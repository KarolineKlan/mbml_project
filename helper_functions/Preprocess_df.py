from sklearn.decomposition import PCA
import torch

# PCA on x_emb_train
def PCA_embedding(x_emb_train, x_emb_test):
    # Fit PCA and choose number of components by covering 95% of variance
    pca = PCA(n_components=0.95)
    pca.fit(x_emb_train.numpy())
    
    # Transform the data
    x_emb_train_pca = torch.tensor(pca.transform(x_emb_train.numpy()), dtype=torch.float32)
    x_emb_test_pca = torch.tensor(pca.transform(x_emb_test.numpy()), dtype=torch.float32)
    return x_emb_train_pca, x_emb_test_pca


def standardize_tensor(tensor, mean=None, std=None):
    if mean is None or std is None:
        # Calculate mean and std if not provided
        mean = torch.mean(tensor)
        std = torch.std(tensor)
    return (tensor - mean) / std, mean, std

def reverse_standardize_tensor(tensor, mean, std):
    return tensor * std + mean


def standardize_input(x_emb_train, x_emb_test, d_demo_train, d_demo_test):
    # Standardize the tensors    
    x_emb_train, mean, std = standardize_tensor(x_emb_train)
    x_emb_test, _, _ = standardize_tensor(x_emb_test, mean, std)
    # Apply PCA to the embeddings
    x_emb_train_pca, x_emb_test_pca = PCA_embedding(x_emb_train, x_emb_test)
    
    # normalize the first and last elements of d_demo_train and d_demo_test aka age and distance
    d_demo_train[:,0], mean, std = standardize_tensor(d_demo_train[:,0])
    d_demo_test[:,0], _, _ = standardize_tensor(d_demo_test[:,0], mean, std)
    d_demo_train[:,-1], mean, std = standardize_tensor(d_demo_train[:,-1])
    d_demo_test[:,-1], _, _ = standardize_tensor(d_demo_test[:,-1], mean, std)

    return x_emb_train_pca, x_emb_test_pca, d_demo_train, d_demo_test
