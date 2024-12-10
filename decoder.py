import numpy as np
from scipy.stats import poisson
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
import requests
import os
import gzip
import shutil
import argparse
import json
from collections import Counter
import time
import threading
import queue
import tkinter as tk
from tkinter import filedialog

def confusion(pos):
    """
    Generates a confusion matrix for each category of image.

    Args:
        pos (np.ndarray): A 2D array of posterior probabilities, 
                            where each column represents an image and 
                            each row represents a category.

    Returns:
        tuple: A tuple containing the confusion matrix (cm) and 
                the indices of the MAP estimates (ind).
    """
    num_images = pos.shape[1]
    num_categories = int(num_images / 100)

    cm = np.zeros((num_categories, num_categories))
    ind = np.zeros(num_images, dtype=int)

    for c in range(num_categories):
        for i in range(100):
            index = c * 100 + i
            mapind = np.argmax(pos[:, index])
            mapcat = mapind // 100  # Use integer division
            cm[c, mapcat] += 1
            ind[index] = mapind

    cm = cm / cm.sum(axis=1, keepdims=True)
    return cm, ind

def linrectify(X):
    """
    Performs linear rectification on input.

    Args:
        X (np.ndarray): Input array.

    Returns:
        np.ndarray: Linearly rectified array.
    """
    return np.maximum(X, 0)

def loglikelihood(activity, image_embeddings, weights, parameters):
    """
    Calculates the log likelihood of the activity given the 
    image embeddings, weights, and parameters.

    Args:
        activity (np.ndarray): A 3D array of spike counts, where 
                                dimensions represent neurons, images, 
                                and repetitions.
        image_embeddings (np.ndarray): A 2D array of image embeddings.
        weights (np.ndarray): A 2D array of synaptic weights.
        parameters (dict): A dictionary of parameters containing 
                            'dt', 'gain', and 'nrep'.

    Returns:
        np.ndarray: A 2D array of log likelihoods.
    """
    # Validate parameters using assertions for brevity
    assert 'dt' in parameters and parameters['dt'] > 0, "Invalid 'dt' parameter"
    assert 'gain' in parameters and parameters['gain'] > 0, "Invalid 'gain' parameter"
    assert 'nrep' in parameters and parameters['nrep'] > 0, "Invalid 'nrep' parameter"

    # Validate input arrays using assertions
    assert isinstance(activity, np.ndarray), "Invalid 'activity' array"
    assert isinstance(image_embeddings, np.ndarray), "Invalid 'image_embeddings' array"
    assert isinstance(weights, np.ndarray), "Invalid 'weights' array"

    meanact = np.ceil(np.mean(activity, axis=2)).astype(int)
    LL = np.zeros((image_embeddings.shape[1], image_embeddings.shape[1]))
    R = rates(image_embeddings, weights, parameters) * parameters['dt']

    # Optimization: Vectorize the inner loop
    A = np.tile(meanact[:, :, np.newaxis], (1, 1, image_embeddings.shape[1]))
    PA = poisson.pmf(A, R[:,:,np.newaxis]) 
    LPA = np.log(PA + 1e-10)  # Add a small constant to avoid log(0)
    LL = np.sum(LPA, axis=0) 

    return LL

def logprior(cp, num_images):
    """
    Returns the log prior for the images, given a specified 
    probability for each category.

    Args:
        cp (np.ndarray): A 1D array of category probabilities.
        num_images (int): The total number of images.

    Returns:
        np.ndarray: A 1D array of log priors.
    """
    assert isinstance(cp, np.ndarray) and np.all(cp > 0) and np.isclose(np.sum(cp), 1.0), "Invalid 'cp' array"

    num_categories = len(cp)
    images_per_category = num_images // num_categories  # Use integer division
    LP = np.zeros(num_images)

    # Optimization: Vectorize the loop
    for c in range(num_categories):
        LP[c * images_per_category:(c + 1) * images_per_category] = np.log(cp[c] / images_per_category)

    return LP

def posaverage(images, pos, navg):
    """
    Returns the average of images weighted by the posterior.

    Args:
        images (np.ndarray): A 2D array of images, where each 
                                column represents an image.
        pos (np.ndarray): A 2D array of posterior probabilities.
        navg (int): The number of images to average.

    Returns:
        np.ndarray: A 2D array of posterior-averaged images.
    """
    assert isinstance(images, np.ndarray), "Invalid 'images' array"
    assert isinstance(pos, np.ndarray) and pos.shape == (images.shape[1], images.shape[1]), "Invalid 'pos' array"
    assert isinstance(navg, int) and 0 < navg <= images.shape[1], "Invalid 'navg' value"

    pa = np.zeros(images.shape)
    ipos = np.argsort(pos, axis=0)[::-1]  # Sort in descending order
    spos = np.take_along_axis(pos, ipos, axis=0)

    for i in range(images.shape[1]):
        topimages = images[:, ipos[:navg, i]]
        pa[:, i] = np.sum(topimages * spos[:navg, i].reshape(1, -1), axis=1)
        pa[:, i] = pa[:, i] / np.max(pa[:, i])  # Normalize

    return pa

def posterior(LL, LP):
    """
    Calculates the posterior probability of each image given 
    the log likelihoods and log priors.

    Args:
        LL (np.ndarray): A 2D array of log likelihoods.
        LP (np.ndarray): A 1D array of log priors.

    Returns:
        np.ndarray: A 2D array of posterior probabilities.
    """
    assert isinstance(LL, np.ndarray) and LL.shape[0] == LL.shape[1], "Invalid 'LL' array"
    assert isinstance(LP, np.ndarray) and LP.shape[0] == LL.shape[1], "Invalid 'LP' array"

    LPOS = LL + LP.reshape(1, -1)  
    POS = np.exp(LPOS - logsumexp(LPOS, axis=0, keepdims=True))
    return POS

def rates(images, weights, parameters):
    """
    Calculates the firing rates of the neurons given the images, 
    weights, and parameters.

    Args:
        images (np.ndarray): A 2D array of images.
        weights (np.ndarray): A 2D array of synaptic weights.
        parameters (dict): A dictionary of parameters containing 
                            'dt' and 'gain'.

    Returns:
        np.ndarray: A 2D array of firing rates.
    """
    assert isinstance(images, np.ndarray), "Invalid 'images' array"
    assert isinstance(weights, np.ndarray), "Invalid 'weights' array"
    assert 'dt' in parameters and parameters['dt'] > 0, "Invalid 'dt' parameter"
    assert 'gain' in parameters and parameters['gain'] > 0, "Invalid 'gain' parameter"

    R = weights @ images
    R = linrectify(R) * parameters['gain']
    return R

def record(images, weights, parameters):
    """
    Simulates an experimental recording session by generating 
    spikes from the neuron population.

    Args:
        images (np.ndarray): A 2D array of images.
        weights (np.ndarray): A 2D array of synaptic weights.
        parameters (dict): A dictionary of parameters containing 
                            'dt', 'gain', and 'nrep'.

    Returns:
        np.ndarray: A 3D array of spike counts.
    """
    assert isinstance(images, np.ndarray), "Invalid 'images' array"
    assert isinstance(weights, np.ndarray), "Invalid 'weights' array"
    assert 'dt' in parameters and parameters['dt'] > 0, "Invalid 'dt' parameter"
    assert 'gain' in parameters and parameters['gain'] > 0, "Invalid 'gain' parameter"
    assert 'nrep' in parameters and parameters['nrep'] > 0, "Invalid 'nrep' parameter"

    activity = np.zeros((weights.shape[0], images.shape[1], parameters['nrep']))

    for n in range(parameters['nrep']):
        activity[:, :, n] = spikes(images, weights, parameters)

    return activity

def show_cm(cm, vis='on'):
    """
    Plots a confusion matrix.

    Args:
        cm (np.ndarray): A 2D array representing the confusion matrix.
        vis (str, optional): Whether to display the plot ('on') or 
                            not ('off'). Defaults to 'on'.
    """
    assert isinstance(cm, np.ndarray) and cm.shape[0] == cm.shape[1], "Invalid 'cm' array"

    num_categories = cm.shape[0]
    plt.figure()
    plt.imshow(100 * cm, cmap='jet')
    plt.xticks(np.arange(num_categories), np.arange(num_categories))
    plt.yticks(np.arange(num_categories), np.arange(num_categories))
    plt.xlabel('Decoded image category')
    plt.ylabel('Presented image category')
    cbar = plt.colorbar()
    cbar.set_label('Categorization frequency (%)')
    plt.set_cmap('jet')

    if vis == 'on':
        plt.show()

def show_images(images, category=None, num_cols=10):
    """
    Displays images from a dataset.

    Args:
        images (np.ndarray): A 2D array of images, where each column 
                                represents an image.
        category (int, optional): If provided, displays images only 
                                    from the specified category. 
                                    Defaults to None.
        num_cols (int, optional): The number of columns in the image 
                                    grid. Defaults to 10.
    """
    assert isinstance(images, np.ndarray), "Invalid 'images' array"

    if category is not None:
        assert isinstance(category, int) and 0 <= category <= (images.shape[1] / 100) - 1, "Invalid 'category' value"
        num_images = 100
        start_index = category * 100
        end_index = (category + 1) * 100
        images = images[:, start_index:end_index]
    else:
        num_images = images.shape[1]

    num_rows = int(np.ceil(num_images / num_cols))
    plt.figure()
    plt.set_cmap('gray')

    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(images[:, i].reshape(int(np.sqrt(images.shape[0])), int(np.sqrt(images.shape[0]))).T)
        plt.axis('equal')
        plt.axis('off')

    if category is not None:
        plt.suptitle(f"Category {category} images")
    else:
        plt.suptitle("All images")
    plt.show()

def show_weights(weights, num_cols=25):
    """
    Displays all the receptive field structures as gray images.

    Args:
        weights (np.ndarray): A 2D array of synaptic weights.
        num_cols (int, optional): The number of columns in the grid of 
                                    receptive fields. Defaults to 25.
    """
    assert isinstance(weights, np.ndarray), "Invalid 'weights' array"

    num_fields = weights.shape[0]
    num_rows = int(np.ceil(num_fields / num_cols))
    plt.figure()
    plt.set_cmap('gray')

    for i in range(num_fields):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(weights[i, :].reshape(int(np.sqrt(weights.shape[1])), int(np.sqrt(weights.shape[1]))).T)
        plt.axis('equal')
        plt.axis('off')

    plt.suptitle("Receptive fields")
    plt.show()

def spikes(images, weights, parameters):
    """
    Generates spikes for the population of neurons based on images 
    and synaptic weights.

    Args:
        images (np.ndarray): A 2D array of images.
        weights (np.ndarray): A 2D array of synaptic weights.
        parameters (dict): A dictionary of parameters containing 
                            'dt' and 'gain'.

    Returns:
        np.ndarray: A 2D array of spike counts.
    """
    assert isinstance(images, np.ndarray), "Invalid 'images' array"
    assert isinstance(weights, np.ndarray), "Invalid 'weights' array"
    assert 'dt' in parameters and parameters['dt'] > 0, "Invalid 'dt' parameter"
    assert 'gain' in parameters and parameters['gain'] > 0, "Invalid 'gain' parameter"

    R = rates(images, weights, parameters)
    S = poisson.rvs(parameters['dt'] * R)
    return S


def show_spikes(activity, parameters):
    """Displays a raster plot of spike activity."""
    plt.figure()
    for neuron_idx in range(activity.shape[0]):
        for rep in range(activity.shape[2]):
            spike_times = np.where(activity[neuron_idx, :, rep] == 1)[0] * parameters['dt']
            plt.plot(spike_times, np.ones_like(spike_times) * neuron_idx, '|k')
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron Index')
    plt.title('Spike Raster Plot')
    plt.show()

def show_spike_histogram(activity, parameters):
    """Displays a histogram of spike counts across neurons."""
    plt.figure()
    all_spike_counts = np.sum(activity, axis=(1, 2))  # Sum across images and repetitions
    plt.hist(all_spike_counts, bins=20)
    plt.xlabel('Spike Count')
    plt.ylabel('Number of Neurons')
    plt.title('Spike Count Histogram')
    plt.show()

def show_spike_rate_plot(activity, parameters):
    """Displays a plot of spike rates over time, averaged across neurons."""
    plt.figure()
    spike_rates = np.mean(activity, axis=0)  # Average across neurons
    time_axis = np.arange(activity.shape[1]) * parameters['dt']
    for rep in range(activity.shape[2]):
        plt.plot(time_axis, spike_rates[:, rep])
    plt.xlabel('Time (s)')
    plt.ylabel('Average Spike Rate')
    plt.title('Average Spike Rate Over Time')
    plt.show()

def show_neuron_heatmap(activity, parameters):
    """Displays a heatmap of spike activity for each neuron across all images and repetitions."""
    plt.figure()
    plt.imshow(np.sum(activity, axis=2), cmap='hot', interpolation='nearest', aspect='auto')  # Sum across repetitions
    plt.xlabel('Image Index')
    plt.ylabel('Neuron Index')
    plt.title('Neuron Spike Activity Heatmap')
    plt.colorbar()
    plt.show()

def load_images(image_paths):
    """
    Loads images from a list of file paths and returns them as a numpy array.

    Args:
        image_paths (list): A list of image file paths.

    Returns:
        np.ndarray: A 2D array of images, where each column represents an image.
    """
    images = []
    for path in image_paths:
        img = Image.open(path).convert('L')
        img_array = np.array(img).flatten()
        images.append(img_array)
    return np.array(images).T

def download_extract_mnist(data_dir):
    """
    Downloads and extracts the MNIST dataset.

    Args:
        data_dir (str): The directory to store the downloaded data.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = ["train-images-idx3-ubyte.gz",
             "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz",
             "t10k-labels-idx1-ubyte.gz",
    ]

    for file in files:
        filepath = os.path.join(data_dir, file)
        if not os.path.exists(filepath):
            print(f"Downloading {file}...")
            try:
                response = requests.get(base_url + file, stream=True)
                response.raise_for_status()
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {file}: {e}")
                # If download fails, try a mirror
                mirror_url = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
                print(f"Trying to download from mirror: {mirror_url + file}")
                try:
                    response = requests.get(mirror_url + file, stream=True)
                    response.raise_for_status()
                    with open(filepath, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                except requests.exceptions.RequestException as e:
                    print(f"Error downloading from mirror: {e}")
                    return

        print(f"Extracting {file}...")
        try:
            with gzip.open(filepath, 'rb') as f_in:
                with open(filepath[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except (gzip.BadGzipFile, OSError) as e:
            print(f"Error extracting {file}: {e}")
            return

def load_mnist_images(data_dir, kind="train"):
    """
    Loads MNIST images from the specified directory.

    Args:
        data_dir (str): The directory containing the MNIST data files.
        kind (str, optional): The type of data to load ('train' or 'test'). 
                            Defaults to 'train'.

    Returns:
        tuple: A tuple containing the images and labels.
    """
    if kind == "train":
        labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte')
        images_path = os.path.join(data_dir, 'train-images-idx3-ubyte')
    elif kind == "test":
        labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte')
        images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte')
    else:
        raise ValueError("Invalid kind argument. Must be 'train' or 'test'.")

    with open(labels_path, 'rb') as lbpath:
        lbpath.read(8)
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        imgpath.read(16)
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784).T

    return images, labels

def load_cifar10_images(data_dir):
    """
    Loads CIFAR-10 images from the specified directory.

    Args:
        data_dir (str): The directory to store the downloaded data.

    Returns:
        tuple: A tuple containing the images and labels.
    """
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    base_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        response = requests.get(base_url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    import tarfile
    print("Extracting CIFAR-10 data...")
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(data_dir)

    images = []
    labels = []
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, "cifar-10-batches-py", f"data_batch_{i}")
        batch_data = unpickle(batch_file)
        images.append(batch_data[b'data'])
        labels.extend(batch_data[b'labels'])

    images = np.concatenate(images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    images = np.array([Image.fromarray(img).convert('L') for img in images])
    images = np.array([np.array(img).flatten() for img in images]).T
    labels = np.array(labels)
    return images, labels

def calculate_category_priors(labels):
    """
    Calculates category priors based on label frequencies.

    Args:
        labels (np.ndarray): A 1D array of labels.

    Returns:
        np.ndarray: A 1D array of category priors.
    """
    label_counts = Counter(labels)
    num_images = len(labels)
    cp = np.array([label_counts[i] / num_images for i in range(len(label_counts))])
    return cp

def run_experiment(images, labels, parameters, spike_formats):
    """
    Runs a single decoding experiment with the given parameters.

    Args:
        images (np.ndarray): A 2D array of images.
        labels (np.ndarray): A 1D array of labels.
        parameters (dict): A dictionary of parameters.
        spike_formats (list): A list of formats for visualizing spike 
                                activity ('raster', 'histogram', 'rate', 
                                'heatmap').
    """
    num_pixels = images.shape[0]
    num_neurons = 500
    weights = np.random.rand(num_neurons, num_pixels)

    cp = calculate_category_priors(labels)

    print("|||-----------------------------------------------------------------")
    print("Beginning decoding experiment:")
    print("|||-----------------------------------------------------------------")

    activity = record(images, weights, parameters)

    for spike_format in spike_formats:
        if spike_format == "raster":
            show_spikes(activity, parameters)
        elif spike_format == "histogram":
            show_spike_histogram(activity, parameters)
        elif spike_format == "rate":
            show_spike_rate_plot(activity, parameters)
        elif spike_format == "heatmap":
            show_neuron_heatmap(activity, parameters)

    ll = loglikelihood(activity, images, weights, parameters)
    lp = logprior(cp, images.shape[1])
    pos = posterior(ll, lp)
    cm, ind = confusion(pos)
    pa = posaverage(images, pos, 10)

    filename = f"figures/case_dt{parameters['dt']:.3f}_gain{parameters['gain']:.1f}_nreps{parameters['nrep']}"
    show_cm(cm, vis='off')
    plt.savefig(f"{filename}.png", dpi=300)
    plt.close()

    show_images(images)
    show_images(pa)

    print("|||-----------------------------------------------------------------")
    print("Finished decoding experiment.")
    print("|||-----------------------------------------------------------------")


def record_live(weights, parameters, input_queue, output_queue):
    """
    Continuously generates spikes for the population of neurons 
    based on images received from the input queue.

    Args:
        weights (np.ndarray): A 2D array of synaptic weights.
        parameters (dict): A dictionary of parameters containing 
                            'dt' and 'gain'.
        input_queue (queue.Queue): A queue to receive input images.
        output_queue (queue.Queue): A queue to send spike data.
    """
    while True:
        images = input_queue.get()  # Get the next image(s) from the queue
        if images is None:
            break  # Sentinel value to stop the thread

        activity = spikes(images, weights, parameters)
        output_queue.put((activity, time.time()))  # Send spike data with timestamp


def run_experiment_live(weights, parameters):
    """
    Runs a live decoding experiment with a GUI for image input.
    """
    input_queue = queue.Queue()
    output_queue = queue.Queue()

    # --- GUI Setup ---
    root = tk.Tk()
    root.title("Live SNN Experiment")

    # Figure for the plot
    fig, ax = plt.subplots()
    line, = ax.plot([], [], '|k')  # For raster plot
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron Index')
    ax.set_title('Live Spike Raster Plot')
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

    # Image frame
    image_frame = tk.Frame(root)
    image_frame.pack()
    image_label = tk.Label(image_frame, text="No Image Loaded")
    image_label.pack()

    # Static input and upload button
    static_input = np.random.rand(784, 1)  # Initial static input
    def upload_image():
        nonlocal static_input
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if file_path:
            # Load and preprocess the image
            image = Image.open(file_path).convert('L')
            image = image.resize((28, 28))  # Resize to match MNIST size
            static_input = np.array(image).flatten().reshape(784, 1)  # Update static input
            image_label.config(text=file_path)
        else:
            image_label.config(text="No Image Loaded")

    upload_button = tk.Button(root, text="Upload Image", command=upload_image)
    upload_button.pack()

    # --- SNN Thread ---
    record_thread = threading.Thread(
        target=record_live, 
        args=(weights, parameters, input_queue, output_queue)
    )
    record_thread.daemon = True
    record_thread.start()

    start_time = time.time()

    def update_plot():
        nonlocal start_time
        try:
            if not output_queue.empty():
                activity, timestamp = output_queue.get()

                # Update the raster plot
                elapsed_time = timestamp - start_time
                for neuron_idx in range(activity.shape[0]):
                    spike_times = np.where(activity[neuron_idx, :] == 1)[0] * parameters['dt']
                    ax.plot(spike_times + elapsed_time,
                            np.ones_like(spike_times) * neuron_idx, '|k')

                ax.relim()
                ax.autoscale_view(True, True, True)
                canvas.draw()

            # Provide input to the SNN
            input_queue.put(static_input) 

        except KeyboardInterrupt:
            print("Stopping experiment...")
            input_queue.put(None)
            record_thread.join()
            root.destroy()  # Close the GUI window
            return
        
        root.after(100, update_plot)  # Update every 100ms

    root.after(100, update_plot)  # Start the plot update loop
    root.mainloop()


def cli_interaction():
    """Provides a command-line interface for user interaction."""
    print("Welcome to the Neural Decoding Experiment!")

    # Dataset selection
    dataset_choice = input("Select dataset (1: MNIST, 2: CIFAR-10, 3: Other): ")
    while dataset_choice not in ["1", "2", "3"]:
        print("Invalid choice. Please enter 1, 2, or 3.")
        dataset_choice = input("Select dataset (1: MNIST, 2: CIFAR-10, 3: Other): ")

    if dataset_choice == "1":
        dataset = "mnist"
        data_dir = "data"
        download_extract_mnist(data_dir)
        images, labels = load_mnist_images(data_dir)
    elif dataset_choice == "2":
        dataset = "cifar10"
        data_dir = "data"
        images, labels = load_cifar10_images(data_dir)
    else:
        dataset = "other"
        image_paths = []  # Replace with your image loading logic
        images = load_images(image_paths)
        labels = []  # Replace with your label loading logic

    # Spike format selection
    spike_formats = []
    while True:
        spike_format = input("Select spike format (1: Raster plot, 2: Histogram, 3: Rate plot, 4: Heatmap, 5: Done): ")
        while spike_format not in ["1", "2", "3", "4", "5"]:
            print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
            spike_format = input("Select spike format (1: Raster plot, 2: Histogram, 3: Rate plot, 4: Heatmap, 5: Done): ")
        if spike_format == "5":
            break
        spike_formats.append({
            "1": "raster",
            "2": "histogram",
            "3": "rate",
            "4": "heatmap"
        }[spike_format])

    # Parameter input
    parameters = {}
    parameters['dt'] = float(input("Enter time step (dt): "))
    parameters['gain'] = float(input("Enter neuron gain: "))
    parameters['nrep'] = int(input("Enter number of repetitions: "))

    # Run the experiment
    run_experiment(images, labels, parameters, spike_formats)


def main():
    parser = argparse.ArgumentParser(description="Run neural decoding experiments.")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "other"], default="mnist", help="Dataset to use (mnist, cifar10 or other)")
    parser.add_argument("--data_dir", default="data", help="Directory to store downloaded data")
    parser.add_argument("--spike_format", choices=["raster", "histogram", "rate", "heatmap"], default="raster", help="Spike visualization format")
    parser.add_argument("--params_file", default="params.json", help="JSON file with experiment parameters")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--live", "-l", action="store_true", help="Run in live mode")
    parser.add_argument("--activate", "-a", action="store_true", help="Activate neural population with custom input")
    args = parser.parse_args()

    if args.interactive:
        cli_interaction()
    else:
        if args.dataset == "mnist":
            download_extract_mnist(args.data_dir)
            images, labels = load_mnist_images(args.data_dir)
        elif args.dataset == "cifar10":
            images, labels = load_cifar10_images(args.data_dir)
        elif args.dataset == "other":
            image_paths = []  # Replace with your image loading logic
            images = load_images(image_paths)
            labels = []  # Replace with your label loading logic

        num_pixels = images.shape[0]
        num_neurons = 500
        weights = np.random.rand(num_neurons, num_pixels)

        with open(args.params_file, 'r') as f:
            params_config = json.load(f)

        parameters = params_config['parameters']
        dts = params_config.get('dts', [0.005, 0.01])
        gains = params_config.get('gains', [0.8, 1.2])
        nreps = params_config.get('nreps', [1, 2])

        cp = calculate_category_priors(labels)

        if args.live:
            print("|||-----------------------------------------------------------------")
            print("Beginning live decoding experiment:")
            print("|||-----------------------------------------------------------------")

            run_experiment_live(weights, parameters)

            print("|||-----------------------------------------------------------------")
            print("Finished live decoding experiment.")
            print("|||-----------------------------------------------------------------")

        elif args.activate:
            print("|||-----------------------------------------------------------------")
            print("Activating neural population with custom input:")
            print("|||-----------------------------------------------------------------")

            run_experiment_activate(weights, parameters)  # Call the new function

            print("|||-----------------------------------------------------------------")
            print("Finished neural population activation.")
            print("|||-----------------------------------------------------------------")

        else:
            print("|||-----------------------------------------------------------------")
            print("Beginning decoding experiments:")
            print("|||-----------------------------------------------------------------")

            for dt in dts:
                for gain in gains:
                    for nrep in nreps:
                        print(f"Running case with dt = {dt:.3f} s, gain = {gain:.1f}, nreps = {nrep}...")
                        print("|------")

                        parameters['dt'] = dt
                        parameters['gain'] = gain
                        parameters['nrep'] = nrep

                        activity = record(images, weights, parameters)

                        spike_formats = [args.spike_format]  # Use the specified format from command-line

                        if args.spike_format == "raster":
                            show_spikes(activity, parameters)
                        elif args.spike_format == "histogram":
                            show_spike_histogram(activity, parameters)
                        elif args.spike_format == "rate":
                            show_spike_rate_plot(activity, parameters)
                        elif args.spike_format == "heatmap":
                            show_neuron_heatmap(activity, parameters)

                        ll = loglikelihood(activity, images, weights, parameters)
                        lp = logprior(cp, images.shape[1])
                        pos = posterior(ll, lp)
                        cm, ind = confusion(pos)
                        pa = posaverage(images, pos, 10)

                        filename = f"figures/case_dt{dt:.3f}_gain{gain:.1f}_nreps{nrep}"
                        show_cm(cm, vis='off')
                        plt.savefig(f"{filename}.png", dpi=300)
                        plt.close()

                        show_images(images)
                        show_images(pa)

            print("|||-----------------------------------------------------------------")
            print("Finished decoding experiments. Figures have been saved in the folder 'figures'")
            print("|||-----------------------------------------------------------------")


def run_experiment_activate(weights, parameters):
    """
    Activates the neural population with custom input and visualizes 
    the spike activity in real-time.

    Args:
        weights (np.ndarray): A 2D array of synaptic weights.
        parameters (dict): A dictionary of parameters.
    """
    input_queue = queue.Queue()
    output_queue = queue.Queue()

    # Start the recording thread
    record_thread = threading.Thread(target=record_live,
                                    args=(weights, parameters,
                                          input_queue, output_queue))
    record_thread.daemon = True
    record_thread.start()

    # Set up the live plot
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], '|k')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron Index')
    ax.set_title('Neural Population Activity')

    start_time = time.time()

    try:
        while True:
            # Get new input images from user
            input_str = input("Enter input values (comma-separated, or 'q' to quit): ")
            if input_str.lower() == 'q':
                break

            try:
                input_values = [float(val.strip()) for val in input_str.split(",")]
                num_pixels = weights.shape[1]  # Get the number of pixels from weights
                new_images = np.array(input_values).reshape(num_pixels, 1)  # Reshape to match the expected format
                input_queue.put(new_images)
            except ValueError:
                print("Invalid input format. Please enter comma-separated numbers.")
                continue

            if not output_queue.empty():
                activity, timestamp = output_queue.get()
                elapsed_time = timestamp - start_time

                # Update the raster plot
                for neuron_idx in range(activity.shape[0]):
                    spike_times = np.where(activity[neuron_idx, :] == 1)[0] * parameters['dt']
                    ax.plot(spike_times + elapsed_time,
                            np.ones_like(spike_times) * neuron_idx, '|k')

                ax.relim()
                ax.autoscale_view(True, True, True)
                fig.canvas.flush_events()

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping experiment...")
    finally:
        input_queue.put(None)
        record_thread.join()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()