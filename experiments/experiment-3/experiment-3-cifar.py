import torch                                                    #type: ignore
import torch.nn as nn                                           #type: ignore
from scipy.stats import kstest                                  #type: ignore
from torchvision import datasets, transforms, utils             #type: ignore
from torch.utils.data import DataLoader                         #type: ignore
import matplotlib.pyplot as plt                                 #type: ignore
import numpy as np                                              #type: ignore
from torchmetrics.image.fid import FrechetInceptionDistance     #type: ignore
from timm.utils import ModelEmaV3                               #type: ignore
import os                                                       #type: ignore
import sys                                                      #type: ignore

sys.path.append("ddpm\\aditey")

from DDPM_cifar import UNET                                     #type: ignore

os.makedirs("graphs-cifar", exist_ok = True)
os.makedirs("samples-cifar", exist_ok = True)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Device: {device}")

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

    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)

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
    n_choice = min(len(samples_np), 50000)
    
    samples_np = np.random.choice(samples_np, size = n_choice, replace = False)
    
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
    plt.title(f"CIFAR Latent T = {num_time_steps}")
    plt.savefig(f"graphs-cifar/hist_T_{num_time_steps}.png", dpi = 300, bbox_inches = 'tight')
    
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
    plt.savefig("graphs-cifar/mean_vs_T.png", dpi = 300, bbox_inches = 'tight')
    plt.close()

    # Variance
    plt.figure()
    plt.plot(T_vals, variances, marker = 'o')
    plt.xlabel("T")
    plt.ylabel("Variance")
    plt.title("Variance vs T")
    plt.savefig("graphs-cifar/variance_vs_T.png", dpi = 300, bbox_inches = 'tight')
    plt.close()

    # KL
    plt.figure()
    plt.plot(T_vals, kls, marker='o')
    plt.xlabel("T")
    plt.ylabel("KL Divergence")
    plt.yscale("log")
    plt.title("KL Divergence vs T")
    plt.savefig("graphs-cifar/kl_vs_T.png", dpi = 300, bbox_inches = 'tight')
    plt.close()

    # Wasserstein
    plt.figure()
    plt.plot(T_vals, wasserstein, marker='o')
    plt.xlabel("T")
    plt.ylabel("Wasserstein Distance")
    plt.yscale("log")
    plt.title("Wasserstein Distance vs T")
    plt.savefig("graphs-cifar/wasserstein_vs_T.png", dpi = 300, bbox_inches = 'tight')
    plt.close()

    # KS
    plt.figure()
    plt.plot(T_vals, ks_stats, marker='o')
    plt.xlabel("T")
    plt.ylabel("KS Statistic")
    plt.yscale("log")
    plt.title("KS Statistic vs T")
    plt.savefig("graphs-cifar/ks_vs_T.png", dpi = 300, bbox_inches='tight')
    plt.close()

    # TV Distanceplt.figure()
    plt.plot(T_vals, tv_vals, marker='o')
    plt.xlabel("T")
    plt.ylabel("Total Variation Distance")
    plt.yscale("log")
    plt.title("TV Distance vs T")
    plt.savefig("graphs-cifar/tv_vs_T.png", dpi = 300, bbox_inches='tight')
    plt.close()

    # Skewness
    plt.figure()
    plt.plot(T_vals, skewness_vals, marker='o')
    plt.xlabel("T")
    plt.ylabel("Skewness")
    plt.title("Skewness vs T")
    plt.savefig("graphs-cifar/skewness_vs_T.png", dpi = 300, bbox_inches='tight')
    plt.close()



'''
Part (b): Finding a relation between image quality (FID) and number of time steps T
'''


def sample(model, scheduler, num_steps, n_samples):

    model.eval()
    
    with torch.no_grad():
    
        img_shape = (n_samples, 3, 32, 32) 
        x = torch.randn(img_shape).to(device)

        total_train_steps = len(scheduler.beta)
        indices = torch.linspace(total_train_steps - 1, 0, num_steps).long()

        for index in range(len(indices)):

            timestep_index = indices[index].item()
            timestep = torch.full((n_samples,), timestep_index, device = device, dtype = torch.long)

            predicted_noise = model(x, timestep)

            alpha_bar_t = scheduler.alpha[timestep_index]
            beta_t = scheduler.beta[timestep_index]
            alpha_t = 1 - beta_t

            coeff = beta_t / torch.sqrt(1 - alpha_bar_t)
            x_prev_mean = (1 / torch.sqrt(alpha_t)) * (x - coeff * predicted_noise)

            if timestep_index > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t) 
                x = x_prev_mean + sigma_t * noise
            else:
                x = x_prev_mean

        return x



def preprocess(image):

    image = (image + 1) / 2
    image = image.clamp(0, 1)
    
    rgb_image = nn.functional.interpolate(image, size = 299)
    rgb_image = (rgb_image * 255).to(torch.uint8)

    return rgb_image



def compute_fid(model, scheduler, dataloader, num_steps, n_samples = 5000):

    fid = FrechetInceptionDistance(feature = 2048).to(device)

    count = 0

    for real_batch, _ in dataloader:
        real_batch = real_batch.to(device)
        
        for i in range(0, real_batch.size(0), 32):
            micro_batch = real_batch[i : i + 32]
            processed_micro = preprocess(micro_batch)
            fid.update(processed_micro, real=True)
            
            count += micro_batch.size(0)

            if count >= n_samples:
                break

        if count >= n_samples:
            break

    batch_size = 32

    for _ in range(0, n_samples, batch_size):
        batch = sample(model, scheduler, num_steps, batch_size)
        
        if _ == 0:
            grid = utils.make_grid(batch[:16], normalize=True, value_range=(-1, 1))
            utils.save_image(grid, f"samples-cifar/cifar_T_{num_steps}.png")
            
        batch = preprocess(batch)
        fid.update(batch, real=False)
        
        torch.cuda.empty_cache()

    return fid.compute().item()



if __name__ == "__main__":

    print("Starting Main (CIFAR-10 Mode)")
    
    torch.manual_seed(42)
    np.random.seed(42)

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(root = "./data", train = True, download = True, transform = transform)
    dataloader = DataLoader(dataset, batch_size = 2048, shuffle = True)

    max_batches = 25
    compiled_list = []
    time_T = [10, 50, 100, 500, 1000, 2000, 5000]

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

    print("Cleaning up memory from Part (a)...")
    if 'sample_points' in locals():
        del sample_points
    if 'compiled_list' in locals():
        del compiled_list

    torch.cuda.empty_cache()
    import gc
    gc.collect()


    print("Part (b) starting")
 
    model = UNET().to(device)
    checkpoint = torch.load("cifar-checkpoints/ddpm_checkpoint_cifar_295_epochs", map_location = device)
    model.load_state_dict(checkpoint['weights'])
    
    if 'ema' in checkpoint:
        ema = ModelEmaV3(model)
        ema.load_state_dict(checkpoint['ema'])
        model = ema.module
    
    model.eval()
    scheduler = DDPM_Scheduler(num_time_steps=1000)

    time_steps = [10, 50, 100, 250, 500, 1000]
    fid_scores = []

    for num_steps in time_steps:
        print(f"Computing FID for sampling steps = {num_steps}...", flush=True)
        fid_score = compute_fid(model, scheduler, dataloader, num_steps, n_samples=5000)
        fid_scores.append((num_steps, fid_score))
        print(f"T = {num_steps} | FID = {fid_score:.4f}", flush=True)

    
    T_vals = [x[0] for x in fid_scores]
    fid_vals = [x[1] for x in fid_scores]
    
    plt.figure()
    plt.plot(T_vals, fid_vals, marker='s', color='r')
    plt.xlabel("Sampling Steps (T)")
    plt.ylabel("FID Score")
    plt.title("CIFAR-10 FID vs Sampling Steps")
    plt.grid()

    plt.savefig("graphs-cifar/fid_vs_T_cifar.png")
    plt.close()


    print("\nFinal Results (CIFAR):")

    for T, fid in fid_scores:
        print(f"T = {T} | FID = {fid:.4f}")