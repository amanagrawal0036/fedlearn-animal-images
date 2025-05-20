import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import time
from pathlib import Path
import math

#Directory Setup
BASE_DIR = Path('PetImages')
CSV_PATH = BASE_DIR / 'labels.csv'
IMAGE_DIR = BASE_DIR / 'train'
IMG_EXT = '.jpg'

#Data configuration and details
IMG_SIZE = 64 
N_CLASSES = 2 
CLASS_MAP = {'Cat': 0, 'Dog': 1}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
TARGET_INITIAL_CLIENT_SIZE = 80

#ML parameters
BATCH_SIZE = 32  
LEARNING_RATE_CML = 0.001
LEARNING_RATE_FL = 0.01 
EPOCHS_CML = 20  
EPOCHS_FL_CLIENT = 20

#FL Simulation
NUM_CLIENTS = 1000 
CLIENT_FRACTION_PER_ROUND = 0.1 
NUM_FL_ROUNDS = 50
INITIAL_SERVER_DATA_FRACTION = 0.05 
INITIAL_CLIENT_DATA_FRACTION = 0.55
VALIDATION_FRACTION = 0.1
NEW_DATA_PER_CLIENT_ROUND = 5

#Using GPU for running because the computations were taking too long
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
SEED = 42

#Seed usage
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#Dataset class for easier usage 
class AnimalDataset(Dataset):
    def __init__(self, csv_path, image_dir, img_ext, class_map, transform=None, data_indices=None):
        self.image_dir = Path(image_dir)
        self.img_ext = img_ext
        self.class_map = class_map
        self.transform = transform

        self._original_data_frame = pd.read_csv(csv_path)
        self._original_data_frame = self._original_data_frame[self._original_data_frame['label'].isin(class_map.keys())]
        if self._original_data_frame['id'].dtype != 'object':
            self._original_data_frame['id'] = self._original_data_frame['id'].astype(str)


        if data_indices is not None:
            valid_indices = [idx for idx in data_indices if idx < len(self._original_data_frame)]
            # Use iloc to select rows based on the provided *original* indices
            self.data_frame = self._original_data_frame.iloc[valid_indices].reset_index(drop=True)
        else:
            self.data_frame = self._original_data_frame.copy().reset_index(drop=True)

    

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Gets data
        img_id = self.data_frame.loc[idx, 'id']
        label_name = self.data_frame.loc[idx, 'label']
        label = self.class_map[label_name]

        #printf("hello 1")
        img_name = self.image_dir / f"{img_id}"

        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
                
        return image, label


#This function loads data
def get_data_loaders(csv_path, image_dir, img_ext, class_map, img_size, batch_size, seed):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Resize
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #Loading the full dataset
    full_dataset_meta = pd.read_csv(csv_path)
    num_total_samples = len(full_dataset_meta)
    
    print(f"Total samples found in CSV: {num_total_samples}")

    #The main dataset object
    full_dataset = AnimalDataset(csv_path, image_dir, img_ext, class_map, transform=transform)

    #split sizes
    num_val_samples = int(VALIDATION_FRACTION * num_total_samples)
    num_train_samples = num_total_samples - num_val_samples

    print(f"Splitting into: training={num_train_samples} and validation={num_val_samples}")

    #split into Train and Validation
    all_indices = list(range(num_total_samples))
    np.random.shuffle(all_indices)
    val_indices = all_indices[:num_val_samples]
    train_indices = all_indices[num_val_samples:]
 
    # Create subset for validation and CML training
    val_dataset = Subset(full_dataset, val_indices)
    cml_train_dataset = Subset(full_dataset, train_indices)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    cml_train_loader = DataLoader(cml_train_dataset, batch_size=batch_size, shuffle=True)

    #Data for FL
    num_server_initial = int(INITIAL_SERVER_DATA_FRACTION * num_train_samples)
    num_clients_initial_pool = int(INITIAL_CLIENT_DATA_FRACTION * num_train_samples)
    num_available_for_new = num_train_samples - num_server_initial - num_clients_initial_pool

    print(f"FL Split: Server Initial={num_server_initial}, Clients Initial Pool Size={num_clients_initial_pool}, Available for New Data={num_available_for_new}")

    np.random.shuffle(train_indices)

    # Assign indices to server, the client pool, and the new data pool
    server_initial_indices = train_indices[:num_server_initial]

    #clients will sample their initial data
    clients_initial_indices_pool = train_indices[num_server_initial : num_server_initial + num_clients_initial_pool]
    new_data_pool_indices = train_indices[num_server_initial + num_clients_initial_pool:]

    # Server's initial dataset
    server_dataset = Subset(full_dataset, server_initial_indices)
    server_loader = DataLoader(server_dataset, batch_size=batch_size, shuffle=True)
    
    #initial data among clients WITHOUT OVERLAP 
    client_datasets = {} 

    if not clients_initial_indices_pool:
        print("Warning: Client initial data pool is empty. All clients will start with no data.")
        for i in range(NUM_CLIENTS):
            client_datasets[i] = []
    else:
        pool_size = len(clients_initial_indices_pool)

        # Calculate how many samples each client can get without overlap
        samples_per_client = pool_size // NUM_CLIENTS
        remaining_samples = pool_size % NUM_CLIENTS
        
        print(f"Each client gets {samples_per_client} indices (no overlap) from a pool of {pool_size} indices.")
        
        random.shuffle(clients_initial_indices_pool)
        start_idx = 0
        for i in range(NUM_CLIENTS):
            end_idx = start_idx + samples_per_client
            if i < remaining_samples:  # Distribute remaining samples
                end_idx += 1
            client_datasets[i] = clients_initial_indices_pool[start_idx:end_idx]
            start_idx = end_idx

        client_data_counts = [len(v) for v in client_datasets.values()]
        print(f"Initial client data sizes: Min={min(client_data_counts)}, Max={max(client_data_counts)}, Avg={np.mean(client_data_counts):.2f}")

    return cml_train_loader, val_loader, server_loader, client_datasets, new_data_pool_indices, full_dataset

#CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=N_CLASSES):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 16x16 -> 8x8

        self._determine_flattened_size(IMG_SIZE)

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def _determine_flattened_size(self, img_size):
        with torch.no_grad():
        
            dummy_input = torch.zeros(1, 3, img_size, img_size)
            x = self.pool1(self.relu1(self.conv1(dummy_input)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.pool3(self.relu3(self.conv3(x)))

            # Calculate the resulting flattened size
            self.flattened_size = x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1) #flatten
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

#Training and Evaluation
def train_model(model, dataloader, criterion, optimizer, device, num_epochs):
    model.train()
    epoch_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        num_batches = 0
        if dataloader and len(dataloader.dataset) > 0:
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, mininterval=1.0)
            for inputs, labels in progress_bar:
                if inputs.shape[0] == 0: continue

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                #printf("Hello 2")

            if num_batches > 0:
                epoch_loss = running_loss / num_batches
                epoch_losses.append(epoch_loss)
            else:
                epoch_losses.append(0.0) 
            epoch_losses.append(0.0) 

    return epoch_losses

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    if not dataloader or not hasattr(dataloader, 'dataset') or len(dataloader.dataset) == 0:
        return 0.0, 0.0

    # print("debug0")
    with torch.no_grad():
        for inputs, labels in dataloader:
            if inputs.shape[0] == 0: continue
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if inputs.size(0) > 0:
                 loss = criterion(outputs, labels)
                 running_loss += loss.item() * inputs.size(0) # Weighted avg
                 _, predicted = torch.max(outputs.data, 1)
                 total += labels.size(0)
                 correct += (predicted == labels).sum().item()
    
    # print("debug1")
    if total == 0:
         return 0.0, 0.0

    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

#FL client simulation
class SimulatedClient:
    def __init__(self, client_id, initial_data_indices):
        self.id = client_id
        self.data_indices = list(initial_data_indices) 
        self.used_new_data_indices = set() #to filter out data which was added to client's after each FL round
        #device parameters
        self.battery_power = random.uniform(20, 100)
        self.comm_type = random.choice(['wifi_strong', 'wifi_weak', 'mobile_paid', 'mobile_free'])
        self.compute_capability = random.uniform(0.5, 1.5)
        
    def add_data(self, new_indices):
        new_unique_indices = [idx for idx in new_indices if idx not in self.used_new_data_indices]
        self.data_indices.extend(new_unique_indices)
        self.used_new_data_indices.update(new_unique_indices)

    def get_dataloader(self, full_dataset, batch_size):
        if not self.data_indices:
            return None 

        client_subset = Subset(full_dataset, self.data_indices)

        if len(client_subset) == 0:
            return None

        actual_batch_size = min(batch_size, len(client_subset))
        return DataLoader(client_subset, batch_size=actual_batch_size, shuffle=True, num_workers=0)

def federated_averaging(global_model_state, client_updates, client_data_sizes):
    if not client_updates:
        return global_model_state

    total_data_size = sum(client_data_sizes)
    if total_data_size == 0:
        print("there's no client data")
        return global_model_state

    first_update_device = next(iter(client_updates[0].values())).device
    aggregated_delta = {name: torch.zeros_like(param_data, device=first_update_device)
                        for name, param_data in client_updates[0].items()}

    # calculating current rnd's weights avg
    for i, delta in enumerate(client_updates):
        if client_data_sizes[i] == 0: continue 
        weight = client_data_sizes[i] / total_data_size
        for name in aggregated_delta:
            if name in delta:
                delta_value = delta[name].to(first_update_device)
                if not delta_value.dtype.is_floating_point:
                    delta_value = delta_value.float()
                
                # weighted cal.
                weighted_delta = delta_value * weight
                if not aggregated_delta[name].dtype.is_floating_point:
                    aggregated_delta[name] = aggregated_delta[name].float()
                
                aggregated_delta[name].add_(weighted_delta)

    # updatin the global (FL server's model) model
    updated_global_state = copy.deepcopy(global_model_state)
    for name in updated_global_state:
        if name in aggregated_delta:
            param_device = updated_global_state[name].device
            # Convert back to original dtype if needed
            if not updated_global_state[name].dtype.is_floating_point:
                aggregated_delta[name] = aggregated_delta[name].to(updated_global_state[name].dtype)
            updated_global_state[name].add_(aggregated_delta[name].to(param_device))

    return updated_global_state

def calculate_model_params_size(model):
    """cost calculation"""
    total_size = 0
    for param in model.parameters():
        if param.requires_grad:
            total_size += param.nelement() * param.element_size()
    return total_size

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Assignment 2 : Federated Learning (Animal Classification) ---")
    print(f"Configuration: {NUM_CLIENTS} clients, {NUM_FL_ROUNDS} FL rounds, {CLIENT_FRACTION_PER_ROUND*100:.1f}% clients per round")
    print(f"Target Initial Client Size (Overlap): {TARGET_INITIAL_CLIENT_SIZE}")

    # 1. Load Data
    print("\n--- 1. Loading Data ---")
    try:
        cml_train_loader, val_loader, server_loader, client_datasets_indices, new_data_pool_indices, full_dataset = \
            get_data_loaders(CSV_PATH, IMAGE_DIR, IMG_EXT, CLASS_MAP, IMG_SIZE, BATCH_SIZE, SEED)
    except Exception as e:
        print(f"error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    #  2. Centralized Machine Learning (CML)
    print("\n--- 2. Centralized Machine Learning (CML) ---")
    print("Using SimpleCNN for CML training.")
    cml_model = SimpleCNN(num_classes=N_CLASSES).to(DEVICE)
    cml_criterion = nn.CrossEntropyLoss()
    cml_optimizer = optim.Adam(cml_model.parameters(), lr=LEARNING_RATE_CML)

    cml_start_time = time.time()
    cml_history = {'loss': [], 'val_loss': [], 'val_acc': []}

    print("Starting CML Training...")
    for epoch in range(EPOCHS_CML):
        epoch_losses = train_model(cml_model, cml_train_loader, cml_criterion, cml_optimizer, DEVICE, num_epochs=1)
        current_loss = epoch_losses[0] if epoch_losses and epoch_losses[0] is not None and not math.isnan(epoch_losses[0]) else 0.0
        cml_history['loss'].append(current_loss)

        val_loss, val_acc = evaluate_model(cml_model, val_loader, cml_criterion, DEVICE)
        cml_history['val_loss'].append(val_loss)
        cml_history['val_acc'].append(val_acc)
        print(f"CML Epoch {epoch+1}/{EPOCHS_CML} => Train Loss: {cml_history['loss'][-1]:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    cml_end_time = time.time()
    cml_training_time = cml_end_time - cml_start_time
    print(f"CML Training finished in {cml_training_time:.2f} seconds.")
    cml_communication_cost = 0
    print(f"CML Communication Cost (during training iterations): {cml_communication_cost}")


    # 3. FL simulation
    print("\n--- 3. Federated Learning ---")
    print("Using SimpleCNN for FL training.")
    fl_model = SimpleCNN(num_classes=N_CLASSES).to(DEVICE)
    fl_criterion = nn.CrossEntropyLoss()

    # building skeleton model
    if server_loader and len(server_loader.dataset) > 0:
        print(f"building 'skeleton' model (SimpleCNN) on server data ({len(server_loader.dataset)} samples)...")
        skeleton_model = SimpleCNN(num_classes=N_CLASSES).to(DEVICE)
        skeleton_optimizer = optim.Adam(skeleton_model.parameters(), lr=LEARNING_RATE_CML)
        num_skeleton_epochs = max(1, EPOCHS_CML // 5)
        train_model(skeleton_model, server_loader, fl_criterion, skeleton_optimizer, DEVICE, num_epochs=num_skeleton_epochs)
        fl_model.load_state_dict(skeleton_model.state_dict())
        print("Skeleton model pre-training finished ")
    else:
        print("No initial server data or server loader is empty. Starting FL with randomly initialized SimpleCNN.")

    # saving starting skeleton model params
    global_model_state = {k: v.cpu() for k, v in fl_model.state_dict().items()}

    # creating clients for FL rounds
    clients = [SimulatedClient(i, client_datasets_indices.get(i, [])) for i in range(NUM_CLIENTS)]
    print(f"Created {len(clients)} simulated clients.")

    current_new_data_pool_indices = list(new_data_pool_indices)

    fl_start_time = time.time()
    fl_history = {'round': [], 'val_loss': [], 'val_acc': [], 'comm_cost_round': [], 'total_comm_cost': []}
    total_fl_communication_cost = 0
    model_param_size_bytes = calculate_model_params_size(fl_model)
    print(f"Estimated size of FL model (SimpleCNN) parameters per transmission: {model_param_size_bytes / 1024:.2f} KB")

    print("Starting FL Training Rounds...")
    for fl_round in range(NUM_FL_ROUNDS):
        round_start_time = time.time()
        print(f"\n--- FL Round {fl_round + 1}/{NUM_FL_ROUNDS} ---")

        # Client Selection
        num_clients_to_select = max(1, int(CLIENT_FRACTION_PER_ROUND * NUM_CLIENTS))
        available_client_indices = list(range(NUM_CLIENTS))
        num_clients_to_select = min(num_clients_to_select, len(available_client_indices))
        if num_clients_to_select == 0:
             print("Warning: No clients available or selected for this round.")
             continue
        
        # randomly selecting clients
        # selected_client_indices = random.sample(available_client_indices, num_clients_to_select)
        
        # Client Selcetion logic based on device params
        client_scores = []
        for i in available_client_indices:
            client = clients[i]
            score = client.battery_power
            if client.comm_type in ['wifi_strong', 'mobile_free']: score += 50
            score += client.compute_capability * 10
            if client.battery_power < 20: score = -1
            client_scores.append((score, i))

        client_scores.sort(key=lambda x: x[0], reverse=True)
        selected_client_indices = [index for score, index in client_scores[:num_clients_to_select]]
        # END REPLACEMENT LINES
        
        selected_clients = [clients[i] for i in selected_client_indices]
        print(f"Selected {len(selected_clients)} clients for this round.")

        client_updates = []
        client_data_sizes = []
        successful_clients_count = 0

        # Client Training Loop
        for client in tqdm(selected_clients, desc="Client Training", leave=False, mininterval=1.0):
            if not client.data_indices: continue 

            local_model = SimpleCNN(num_classes=N_CLASSES).to(DEVICE)
            local_model.load_state_dict({k: v.to(DEVICE) for k, v in global_model_state.items()})
            
            client_dataloader = client.get_dataloader(full_dataset, BATCH_SIZE)
            if client_dataloader is None or len(client_dataloader.dataset) == 0 :
                continue 

            # Local training
            local_optimizer = optim.SGD(local_model.parameters(), lr=LEARNING_RATE_FL)
            train_model(local_model, client_dataloader, fl_criterion, local_optimizer, DEVICE, num_epochs=EPOCHS_FL_CLIENT)

            # saving local model params
            local_state = local_model.state_dict()
            delta = {name: local_state[name].cpu() - global_model_state[name].cpu()
                    for name in global_model_state}

            client_updates.append(delta)
            client_data_sizes.append(len(client_dataloader.dataset))
            successful_clients_count += 1

            # adding new data for subsequent rounds
            if current_new_data_pool_indices:
                # Calculate how many new samples we can give each client without overlap
                num_new_per_client = min(NEW_DATA_PER_CLIENT_ROUND, 
                                       len(current_new_data_pool_indices) // successful_clients_count) if successful_clients_count > 0 else 0
                
                if num_new_per_client > 0:
                    random.shuffle(current_new_data_pool_indices)
                
                    start_idx = 0
                    for client in selected_clients:
                        if start_idx + num_new_per_client > len(current_new_data_pool_indices):
                            break
                        new_indices_for_client = current_new_data_pool_indices[start_idx:start_idx + num_new_per_client]
                        client.add_data(new_indices_for_client)
                        start_idx += num_new_per_client
                    
                    current_new_data_pool_indices = current_new_data_pool_indices[start_idx:]

        # updating server model
        if client_updates:
            global_model_state = federated_averaging(global_model_state, client_updates, client_data_sizes)
            fl_model.load_state_dict({k: v.to(DEVICE) for k, v in global_model_state.items()})
        else:
            print("No client updates received in this round.")

        # model evaluation
        val_loss, val_acc = evaluate_model(fl_model, val_loader, fl_criterion, DEVICE)

        # cost of communicatn
        round_comm_cost = model_param_size_bytes * successful_clients_count * 2
        total_fl_communication_cost += round_comm_cost

        fl_history['round'].append(fl_round + 1)
        fl_history['val_loss'].append(val_loss)
        fl_history['val_acc'].append(val_acc)
        fl_history['comm_cost_round'].append(round_comm_cost)
        fl_history['total_comm_cost'].append(total_fl_communication_cost)

        round_end_time = time.time()
        print(f"FL Round {fl_round + 1} ({round_end_time - round_start_time:.2f}s) => Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    fl_end_time = time.time()
    fl_training_time = fl_end_time - fl_start_time
    print(f"\nFL Training finished in {fl_training_time:.2f} seconds.")
    print(f"Total FL Communication Cost: {total_fl_communication_cost / (1024 * 1024):.3f} MB")


    # Final Summaries
    print("\nCommunication Cost Summary:")
    print(f"  - Centralized ML (CML - ComplexCNN): {cml_communication_cost / (1024*1024):.3f} MB (during training iterations)")
    final_fl_comm_cost_mb = total_fl_communication_cost / (1024*1024)
    print(f"  - Federated ML (FL - SimpleCNN): {final_fl_comm_cost_mb:.3f} MB (over {NUM_FL_ROUNDS} rounds)")

    final_cml_acc = cml_history['val_acc'][-1] if cml_history['val_acc'] else 'N/A'
    final_fl_acc = fl_history['val_acc'][-1] if fl_history['val_acc'] else 'N/A'
    print("\nFinal Accuracy Summary:")
    try:
        print(f"  - CML (ComplexCNN) Final Validation Accuracy: {final_cml_acc:.2f}% (after {EPOCHS_CML} epochs)")
    except (TypeError, ValueError):
        print(f"  - CML (ComplexCNN) Final Validation Accuracy: {final_cml_acc} (after {EPOCHS_CML} epochs)")
    try:
         print(f"  - FL (SimpleCNN) Final Validation Accuracy: {final_fl_acc:.2f}% (after {NUM_FL_ROUNDS} rounds)")
    except (TypeError, ValueError):
         print(f"  - FL (SimpleCNN) Final Validation Accuracy: {final_fl_acc} (after {NUM_FL_ROUNDS} rounds)")

    print("\n--- Comparison Complete ---")