'''
Experiment-3:

How does the value of T, the number of timesteps, impact:
• The distribution of the latent variable x_T and the extent to which x_T approaches a
Gaussian distribution.
• The image quality, and can we empirically obtain a relation between T and FID scores?

This experiment aims to find answers to the above to questions.
'''


import torch                                                    #type: ignore
import torch.nn as nn                                           #type: ignore
from scipy.stats import kstest                                  #type: ignore
from torchvision import datasets, transforms                    #type: ignore
from torch.utils.data import DataLoader                         #type: ignore
import matplotlib.pyplot as plt                                 #type: ignore
import numpy as np                                              #type: ignore
from torchmetrics.image.fid import FrechetInceptionDistance     #type: ignore
import open_clip                                                #type: ignore
import sys

sys.path.append(r"ddpm/aditey")
sys.path.append(r"experiments\\experiment-3")

from DDPM_mnist import UNET                                     #type: ignore


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu") 


'''
Part (a): Validating whether the distribution of latent variable x_T approaches gaussian as T increases.
'''

class DDPM_Scheduler(nn.Module):

    def __init__(self, num_time_steps: int = 1000):
    
        super().__init__()

        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad = False).to(device)

        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim = 0).requires_grad_(False).to(device)

    def forward(self, timestep):

        return self.alpha[timestep], self.beta[timestep]
    



def latent_distribution(dataset, num_time_steps, max_batch_size):

    scheduler = DDPM_Scheduler(num_time_steps = num_time_steps)
    alpha_bar_T = scheduler.alpha[-1]

    sample_points = []

    for index, (point, _) in enumerate(dataset):
        
        point = point.to(device)

        noise = torch.randn_like(point)
        sampling_point = (alpha_bar_T.sqrt() * point + (1 - alpha_bar_T).sqrt() * noise)

        sample_points.append(sampling_point.cpu())

        if index >= max_batch_size:
            break

    sample_points = torch.cat(sample_points, dim = 0)

    return sample_points



def distribution_stats(sample_points):

    sample_points_flat = sample_points.view(-1)

    mean = torch.mean(sample_points_flat)
    variance = torch.var(sample_points_flat)

    return mean, variance



def kl_divergence_gaussian(mean, variance):

    return 0.5 * (mean**2 + variance - 1 - torch.log(variance))



def wasserstein_distance(mean, variance):

    standard_deviation = torch.sqrt(variance)

    return torch.sqrt(mean**2 + (standard_deviation - 1)**2)



def kolmogorov_smirnov(samples):

    samples_np = samples.view(-1).cpu().numpy()
    samples_np = np.random.choice(samples_np, size = 50000, replace = False)
    statistics, p_value = kstest(samples_np, 'norm')

    return statistics, p_value



def skewness(samples, mean, variance):

    samples = samples.view(-1)
    standard_deviation = torch.sqrt(variance)

    return torch.mean(((samples - mean)/standard_deviation)**3)



def total_variation_distance(samples, bins = 100):

    samples = samples.view(-1).cpu().numpy()

    hist, bin_edges = np.histogram(samples, bins = bins, density = True)

    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    gaussian = (1/np.sqrt(2*np.pi)) * np.exp(-centers**2 / 2)

    tv = 0.5 * np.sum(np.abs(hist - gaussian)) * (centers[1] - centers[0])

    return tv



def histogram(sample_points, num_time_steps):

    sample_points_flat = sample_points.view(-1).cpu().numpy()

    plt.hist(sample_points_flat, bins = 1000, density = True)

    x_axis = np.linspace(-4, 4, 1000)

    plt.plot(x_axis, (1/np.sqrt(2*np.pi))*np.exp(-x_axis**2/2))
    plt.title(f"T = {num_time_steps}")
    
    plt.savefig(f"experiments/experiment-3/graphs/hist_T_{num_time_steps}.png", dpi = 300, bbox_inches = 'tight')
    plt.close()



def plot_metrics(compiled_list):


    T_vals = [x[0] for x in compiled_list]

    means = [x[1].item() for x in compiled_list]
    variances = [x[2].item() for x in compiled_list]
    kls = [x[3].item() for x in compiled_list]
    wasserstein = [x[4].item() for x in compiled_list]
    ks_stats = [x[5][0] for x in compiled_list]
    skewness_vals = [x[6].item() for x in compiled_list]
    tv_vals = [x[7] for x in compiled_list]

    # Mean
    plt.figure()
    plt.plot(T_vals, means, marker = 'o')
    plt.xlabel("T")
    plt.ylabel("Mean")
    plt.title("Mean vs T")
    plt.savefig("experiments/experiment-3/graphs/mean_vs_T.png", dpi = 300, bbox_inches = 'tight')
    plt.close()

    # Variance
    plt.figure()
    plt.plot(T_vals, variances, marker = 'o')
    plt.xlabel("T")
    plt.ylabel("Variance")
    plt.title("Variance vs T")
    plt.savefig("experiments/experiment-3/graphs/variance_vs_T.png", dpi = 300, bbox_inches = 'tight')
    plt.close()

    # KL
    plt.figure()
    plt.plot(T_vals, kls, marker='o')
    plt.xlabel("T")
    plt.ylabel("KL Divergence")
    plt.yscale("log")
    plt.title("KL Divergence vs T")
    plt.savefig("experiments/experiment-3/graphs/kl_vs_T.png", dpi = 300, bbox_inches = 'tight')
    plt.close()

    # Wasserstein
    plt.figure()
    plt.plot(T_vals, wasserstein, marker='o')
    plt.xlabel("T")
    plt.ylabel("Wasserstein Distance")
    plt.yscale("log")
    plt.title("Wasserstein Distance vs T")
    plt.savefig("experiments/experiment-3/graphs/wasserstein_vs_T.png", dpi = 300, bbox_inches = 'tight')
    plt.close()

    # KS
    plt.figure()
    plt.plot(T_vals, ks_stats, marker='o')
    plt.xlabel("T")
    plt.ylabel("KS Statistic")
    plt.yscale("log")
    plt.title("KS Statistic vs T")
    plt.savefig("experiments/experiment-3/graphs/ks_vs_T.png", dpi = 300, bbox_inches='tight')
    plt.close()

    # TV Distanceplt.figure()
    plt.plot(T_vals, tv_vals, marker='o')
    plt.xlabel("T")
    plt.ylabel("Total Variation Distance")
    plt.yscale("log")
    plt.title("TV Distance vs T")
    plt.savefig("experiments/experiment-3/graphs/tv_vs_T.png", dpi = 300, bbox_inches='tight')
    plt.close()

    # Skewness
    plt.figure()
    plt.plot(T_vals, skewness_vals, marker='o')
    plt.xlabel("T")
    plt.ylabel("Skewness")
    plt.title("Skewness vs T")
    plt.savefig("experiments/experiment-3/graphs/skewness_vs_T.png", dpi = 300, bbox_inches='tight')
    plt.close()



'''
Part (b): Finding a relation betwene image quality (FID, Clip MMD etc.) and number of time steps T
'''


def sample(model, scheduler, num_steps, n_samples):

    sample_image = torch.randn(n_samples, 1, 28, 28).to(device)

    total_timesteps = scheduler.beta.shape[0] - 1
    timesteps = torch.linspace(total_timesteps, 0, num_steps).long()

    for timestep in timesteps:

        timestep = int(timestep.item())

        predicted_noise = model(sample_image, timestep)

        alpha_t = scheduler.alpha[timestep]
        beta_t = scheduler.beta[timestep]

        sample_image = (1 / torch.sqrt(alpha_t)) * (sample_image - ((1 - alpha_t) / torch.sqrt(1 - scheduler.alpha[timestep])) * predicted_noise)

        if timestep > 0:
            noise = torch.randn_like(sample_image)
            sample_image = sample_image + torch.sqrt(beta_t) * noise

    return sample_image


def preprocess(image):

    image = (image + 1) / 2
    image = image.clamp(0, 1)

    rgb_image = image.repeat(1, 3, 1, 1)
    rgb_image = nn.functional.interpolate(rgb_image, size = 299)

    return rgb_image


def compute_fid(model, scheduler, dataloader, num_steps, n_samples=5000):

    fid = FrechetInceptionDistance(feature=64).to(device)

    count = 0
    for real_batch, _ in dataloader:

        real_batch = real_batch.to(device)
        real_batch = preprocess(real_batch)

        fid.update(real_batch, real=True)

        count += real_batch.shape[0]
        if count >= n_samples:
            break

    fake_images = sample(model, scheduler, num_steps, n_samples)
    fake_images = preprocess(fake_images)

    fid.update(fake_images, real = False)

    return fid.compute().item()


def get_clip_embeddings(images):

    images = (images + 1) / 2
    images = images.clamp(0, 1)

    images = images.repeat(1, 3, 1, 1)
    images = nn.functional.interpolate(images, size = 224)

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1,3,1,1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1,3,1,1)

    images = (images - mean) / std

    with torch.no_grad():
        features = clip_model.encode_image(images)
        features = nn.functional.normalize(features, dim = -1)

    return features


def rbf_kernel(x, y, sigma = 1.0):

    x_norm = (x**2).sum(dim=1).view(-1, 1)
    y_norm = (y**2).sum(dim=1).view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())

    kernel = torch.exp(-dist / (2 * sigma**2))

    return kernel


def compute_cmmd(real_images, fake_images, sigma = 1.0):

    real_features = get_clip_embeddings(real_images)
    fake_features = get_clip_embeddings(fake_images)

    K_xx = rbf_kernel(real_features, real_features, sigma)
    K_yy = rbf_kernel(fake_features, fake_features, sigma)
    K_xy = rbf_kernel(real_features, fake_features, sigma)

    m = real_features.shape[0]
    n = fake_features.shape[0]

    mmd = (
        (K_xx.sum() - torch.diag(K_xx).sum()) / (m * (m - 1)) +
        (K_yy.sum() - torch.diag(K_yy).sum()) / (n * (n - 1)) -
        2 * K_xy.mean()
    )

    return mmd.item()


def compute_cmmd_metric(model, scheduler, dataloader, num_steps, n_samples = 2000):

    real_images_list = []
    
    count = 0
    for real_batch, _ in dataloader:

        real_batch = real_batch.to(device)
        real_images_list.append(real_batch)

        count += real_batch.shape[0]
        if count >= n_samples:
            break

    real_images = torch.cat(real_images_list, dim=0)[:n_samples]

    with torch.no_grad():
        fake_images = sample(model, scheduler, num_steps, n_samples)

    with torch.no_grad():    
        cmmd_score = compute_cmmd(real_images, fake_images)

    return cmmd_score



'''
Executing Code
'''


if __name__ == "__main__":

    torch.manual_seed(42)
    np.random.seed(42)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    dataset = datasets.MNIST(root = "./data", train = True, download = True, transform = transform)
    dataloader = DataLoader(dataset, batch_size = 4096, shuffle = True)

    # Part(a)

    max_batches = 100

    compiled_list = []

    time_T = [10, 50, 100, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000]

    for num_time_steps in time_T:

        sample_points = latent_distribution(dataloader, num_time_steps, max_batches)
        mean, variance = distribution_stats(sample_points)

        kl_divergence = kl_divergence_gaussian(mean, variance)
        wasserstein_dist = wasserstein_distance(mean, variance)
        kolmogorov_smirnov_test = kolmogorov_smirnov(sample_points)
        skewness_value = skewness(sample_points, mean, variance)
        tv_dist = total_variation_distance(sample_points)

        combined_tuple = (num_time_steps, mean, variance, kl_divergence, wasserstein_dist, kolmogorov_smirnov_test, skewness_value, tv_dist)
        compiled_list.append(combined_tuple)

        histogram(sample_points, num_time_steps)

        print(f"For T: {num_time_steps} | KL Divergence: {kl_divergence.item():.4f} | "
              f"Wasserstein Distance: {wasserstein_dist.item():.4f} | KS Test: {kolmogorov_smirnov_test[0]:.4f} | "
              f"TV distance: {tv_dist} | Skewness: {skewness_value.item():.4f}")


    plot_metrics(compiled_list)


    # Part(b)

    dataloader = DataLoader(dataset, batch_size = 256, shuffle = False)

    model = UNET().to(device)
    model.load_state_dict(torch.load("ddpm/aditey/mnist-checkpoints/ddpm_checkpoint.pth", map_location = device, pickle_module=__import__('pickle')))
    model.eval()

    scheduler = DDPM_Scheduler(num_time_steps = 1000)

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained = "openai")
    clip_model = clip_model.to(device)
    clip_model.eval()

    time_steps = [10, 20, 50, 100, 200, 500, 1000]

    fid_scores = []
    cmmd_scores = []

    for num_steps in time_steps:

        print(f"Computing FID for steps = {num_steps}")

        fid_score = compute_fid(model, scheduler, dataloader, num_steps)
        fid_scores.append((num_steps, fid_score))

        cmmd = compute_cmmd_metric(model, scheduler, dataloader, num_steps)
        cmmd_scores.append((num_steps, cmmd))

        print(f"T = {num_steps} | FID = {fid_score:.4f}")
        print(f"T = {num_steps} | CMMD = {cmmd:.6f}")

    
    T_vals = [x[0] for x in fid_scores]
    fid_vals = [x[1] for x in fid_scores]

    plt.figure()
    plt.plot(T_vals, fid_vals, marker = 'o')
    plt.xlabel("Sampling Steps (T)")
    plt.ylabel("FID Score")
    plt.title("FID vs Sampling Steps")
    plt.grid()

    plt.savefig("experiments/experiment-3/graphs/fid_vs_T.png", dpi = 300, bbox_inches = 'tight')
    plt.close()


    T_vals = [x[0] for x in cmmd_scores]
    cmmd_vals = [x[1] for x in cmmd_scores]

    plt.figure()
    plt.plot(T_vals, cmmd_vals, marker = 'o')
    plt.xlabel("Sampling Steps (T)")
    plt.ylabel("CMMD Score")
    plt.title("CMMD vs Sampling Steps")
    plt.grid()

    plt.savefig("experiments/experiment-3/graphs/cmmd_vs_T.png", dpi = 300, bbox_inches = 'tight')
    plt.close()


    print("\nFinal Results:")
    for T, fid in fid_scores:
        print(f"T = {T} | FID = {fid:.4f}")

    for T, cmmd in cmmd_scores:
        print(f"T = {T} | CMMD = {cmmd:.6f}")
        
