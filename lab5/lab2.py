import torch
import model
import loader
import argparse
from tqdm import tqdm
import time
import math
import numpy as np

from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-c', '--device', default = 'cuda')
parser.add_argument('-w', '--num-of-workers', default = 2)
parser.add_argument('-d', '--dataset-path', default = '../dataset')
parser.add_argument('-o', '--optimizer', default = 'sgd')
parser.add_argument('-e', '--epochs', default = '5')
parser.add_argument('-b', '--batch-size', default = 128)
parser.add_argument('-pe', '--print-epochs', default = 'true')
parser.add_argument('-dd', '--download-data', default = 'true')
parser.add_argument('-rc3', '--run-code-3', default = 'false')
parser.add_argument('-p', '--run-profiler', default = 'false')
parser.add_argument('-nbn', '--no-batch-norm', default = 'false')
parser.add_argument('-mnw', '--max-num-of-workers-for-code-3', default = 8)
parser.add_argument('-tfn', '--trace-file-name', default = 'trace')


args = parser.parse_args()
print("...........................running new python file.............................")
print(
    f"python3 lab2.py --device={args.device} \
    --dataset-path={args.dataset_path} \
    --optimizer={args.optimizer} \
    --num-of-workers={args.num_of_workers} \
    --epochs={args.epochs} \
    --batch-size={args.batch_size} \
    --print-epochs={args.print_epochs} \
    --download-data={args.download_data} \
    --run-code-3={args.run_code_3} \
    --run-profiler={args.run_profiler} \
    --no-batch-norm={args.no_batch_norm} \
    --max-num-of-workers-for-code-3={args.max_num_of_workers_for_code_3}"
)

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_gradients(model): 
    return sum(p.grad.numel() for p in model.parameters() if p.grad is not None)


def print_model(device):
    
    ResNet_Model = None

    if device == 'cpu' or (device == 'cuda' and torch.cuda.is_available()):
        ResNet_Model = model.ResNet18().to(device)
    elif device == 'cuda' and torch.cuda.is_available() == False:
        print("[ ! ] No cuda device found")
    else:
        print("[ ! ] Either specify 'cuda' OR 'cpu'")
    
    print(ResNet_Model)
    print(count_parameters(ResNet_Model)) 
    return 
 
def test_loader(dataset_root_path, batch_size, number_of_workers):
    
    Loader = loader.Torch_DataLoader(dataset_root_path, batch_size, number_of_workers)

    print(Loader)
    return 
    
def train_model(dataset_root_path, batch_size, number_of_workers, device, optimizer, epochs, print_epochs, download_data, no_batch_norm):

    Loader = loader.Torch_DataLoader(dataset_root_path, batch_size, number_of_workers, download_data)
    
    Model = torch.nn.DataParallel( model.ResNet18(use_batch_norm = not no_batch_norm).to(device), device_ids = [i for i in range(torch.cuda.device_count())])

    Loss_Function = nn.CrossEntropyLoss()
   
    
    Optimizer = None

    if optimizer == 'sgd':
        Optimizer = torch.optim.SGD(
            Model.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 5e-4, nesterov = False
        )
    elif optimizer == 'sgd-nesterov':
        Optimizer = torch.optim.SGD(
            Model.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 5e-4, nesterov = True
        )
        print("Using SGD-NESTROV")
    elif optimizer == 'adadelta':
        Optimizer = torch.optim.Adadelta(
            Model.parameters(), lr = 0.1, weight_decay = 5e-4
        )
        print("Using ADA-DELTA")
    elif optimizer == 'adagrad':
        Optimizer = torch.optim.Adagrad(
            Model.parameters(), lr = 0.1, weight_decay = 5e-4
        )
        print("Using ADA-GRAD")
    elif optimizer == 'adam':
        Optimizer = torch.optim.Adam(
            Model.parameters(), lr = 0.1, weight_decay = 5e-4
        )
        print("Using Adam")
    else:
        print(f"No optimizer by name {optimizer}")
        print( 
            "Available options for '--optimizer' are 'sgd', 'adadelta', 'adagrad', 'adam' and 'sgd-nesterov' "
        )
    
    total_iters_per_epoch = len(Loader)
    load_times = []
    train_times = []
    avg_epoch_time = []
    
    print("Number of trainable parameters : ", count_parameters(Model)) 

    for epoch in range(epochs):

        Model.train()

        total_predictions = 0
        correct_predictions = 0
        batch_len = 0
        train_loss = 0

        t_load = 0
        t_train = 0
        t_cpu_gpu = 0
        count_steps = 0
        top_1_label = [0] * 10
        expected_labels = [0] * 10
        top_1_label = np.array(top_1_label) 
        expected_labels = np.array(expected_labels)
        
        if print_epochs == True:
            print("-----------------------------------------------")
            print("Epoch : ", epoch + 1)
        epoch_start_time = time.time()
        t_2 = time.time()
        for i, (x, y) in enumerate(tqdm(Loader, disable = True)):
            
            t_load += (time.time() - t_2) if i > 0 else 0
            t_3 = time.time()
            x, y = x.to(device), y.to(device)
           
            t_0 = time.time()
            t_cpu_gpu += t_0 - t_3
            
            Optimizer.zero_grad()
            output = Model(x)
            loss = Loss_Function(output, y) 
            loss.backward()
            Optimizer.step()
            
            torch.cuda.synchronize()
            t_train += time.time() - t_0

            _, predicted = output.max(1)
            total_predictions += y.size(0)
            right_predictions = predicted.eq(y)
            for j, pred in enumerate(right_predictions):
                if pred == True:
                    top_1_label[predicted[j]] += 1
            # expected_labels[y[j]] += 1        

            correct_predictions += right_predictions.sum().item()
            train_loss += loss.item()

            t_2 = time.time()
            count_steps += 1
           
        full_epoch_time = time.time() - epoch_start_time
        if print_epochs == True:
            print("Time taken for Epoch : ", '%.4f'%(full_epoch_time), "sec") 
            print("Load Time for epoch : ", '%.6f'%(t_load), "sec with", number_of_workers, "workers", " | cpu to gpu time : ", '%.6f'%(t_cpu_gpu), "| Train Time : ", '%.6f'%(t_train), "sec | steps in epoch : ", count_steps)   
            print("Avg Loss : ", '%.3f'%(train_loss / total_iters_per_epoch), " | Avg Accuracy : ", (correct_predictions * 100) / total_predictions, "%")
            print("Top 1 accuracy is ", np.argmax(top_1_label), "among the individual label accuracies",  (top_1_label / 5000) * 100)
        
        load_times.append(t_load)
        train_times.append(t_train)
        avg_epoch_time.append(full_epoch_time) 

    # print(torch.cuda.memory_summary(device=device))    
    print("Number of gradiants : ", count_gradients(Model))
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)) 
    return (sum(load_times)/epochs, sum(train_times)/epochs, sum(avg_epoch_time)/epochs)

def train_model_with_profiler(dataset_root_path, batch_size, number_of_workers, device, optimizer, epochs, print_epochs, download_data, trace_file_name):

    Loader = loader.Torch_DataLoader(dataset_root_path, batch_size, number_of_workers, download_data)
    
    Model = model.ResNet18().to(device)

    Loss_Function = nn.CrossEntropyLoss()
   
    
    Optimizer = None

    if optimizer == 'sgd':
        Optimizer = torch.optim.SGD(
            Model.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 5e-4, nesterov = False
        )
    elif optimizer == 'sgd-nesterov':
        Optimizer = torch.optim.SGD(
            Model.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 5e-4, nesterov = True
        )
    elif optimizer == 'adadelta':
        Optimizer = torch.optim.Adadelta(
            Model.parameters(), lr = 0.1, weight_decay = 5e-4
        )
    elif optimizer == 'adagrad':
        Optimizer = torch.optim.Adagrad(
            Model.parameters(), lr = 0.1, weight_decay = 5e-4
        )
    elif optimizer == 'adam':
        Optimizer = torch.optim.Adam(
            Model.parameters(), lr = 0.1, weight_decay = 5e-4
        )
    else:
        print(f"No optimizer by name {optimizer}")
        print( 
            "Available options for '--optimizer' are 'sgd', 'adadelta', 'adagrad', 'adam' and 'sgd-nesterov' "
        )
    
    total_iters_per_epoch = len(Loader)
    load_times = []
    train_times = []
    avg_running_time = []
    
    
    for epoch in range(epochs):

        Model.train()

        total_predictions = 0
        correct_predictions = 0
        batch_len = 0
        train_loss = 0

        t_load = 0
        t_train = 0
        top_1_label = [0] * 10
        expected_labels = [0] * 10
        top_1_label = np.array(top_1_label) 
        expected_labels = np.array(expected_labels)
        
        if print_epochs == True:
            print("-----------------------------------------------")
            print("Epoch : ", epoch + 1)
        epoch_start_time = time.time()
        t_2 = time.time()
#        print_out_till = math.inf
        profile_window = True
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
            for i, (x, y) in enumerate(tqdm(Loader, disable = True)):
                
                t_load += (time.time() - t_2) 
                x, y = x.to(device), y.to(device)
                
                t_0 = time.time()
                
                with record_function("Training"):
                    Optimizer.zero_grad()
                    output = Model(x)
                    loss = Loss_Function(output, y) 
                    loss.backward()
                    Optimizer.step()
                    torch.cuda.synchronize()

                t_train += time.time() - t_0
                with record_function("Calculating Metrics"): 
                    _, predicted = output.max(1)
                    total_predictions += y.size(0)
                    right_predictions = predicted.eq(y)
                    for j, pred in enumerate(right_predictions):
                        if pred == True:
                            top_1_label[predicted[j]] += 1
                        # expected_labels[y[j]] += 1        

                    correct_predictions += right_predictions.sum().item()
                    train_loss += loss.item()
                if i == 9 and profile_window == True:
                    break
                t_2 = time.time()
            
            full_epoch_time = time.time() - epoch_start_time
            if print_epochs == True:
                print("Time taken for Epoch : ", '%.4f'%(full_epoch_time), "sec") 
                print("Load Time for epoch : ", '%.6f'%(t_load), "sec with", number_of_workers, "workers", " | g Train Time : ", '%.6f'%(t_train), "sec")   
                print("Avg Loss : ", '%.3f'%(train_loss / total_iters_per_epoch), " | Avg Accuracy : ", (correct_predictions * 100) / total_predictions, "%")
                print("Top 1 accuracy is ", np.argmax(top_1_label), "among the individual label accuracies",  (top_1_label / 5000) * 100)
                
            load_times.append(t_load)
            train_times.append(t_train)
            avg_running_time.append(full_epoch_time)
        
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)) 
    prof.export_chrome_trace(f"{trace_file_name}.json")
    return (sum(load_times)/epochs, sum(train_times)/epochs, sum(avg_running_time)/epochs)


def main():
    
    Torch_Device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Torch Device : {Torch_Device}")

    ResNet_Model = model.ResNet18().to(Torch_Device)

    print(ResNet_Model)


if __name__ == "__main__":
#    print("[ . ] __main__()")
#    print_model(args.device)
    # test_loader(args.dataset_path, 128, args.num_of_workers)
    
    if args.run_code_3 == 'false' and args.run_profiler == 'false':
        load_times, train_times, epoch_times = train_model(
            dataset_root_path = args.dataset_path, 
            batch_size = int(args.batch_size), 
            number_of_workers = int(args.num_of_workers), 
            device = args.device, 
            optimizer = args.optimizer, 
            epochs = int(args.epochs),
            print_epochs = True if args.print_epochs == 'true' else False,
            download_data = True if args.download_data == 'true' else False,
            no_batch_norm = True if args.no_batch_norm == 'true' else False
        )

        print(f"Avg Data Load times with `{args.num_of_workers}` workers is `{'%.6f'%load_times}` sec and Avg Train Times is `{'%.6f'%train_times}` sec")
        print(f"Avg epoch time is `{'%.4f'%epoch_times}` sec")
    
    if args.run_profiler == 'true':
        
        _, _, _ = train_model_with_profiler(
            dataset_root_path = args.dataset_path, 
            batch_size = int(args.batch_size), 
            number_of_workers = int(args.num_of_workers), 
            device = args.device, 
            optimizer = args.optimizer, 
            epochs = 1, 
            print_epochs = False,
            download_data = True if args.download_data == 'true' else False,
            trace_file_name = args.trace_file_name
        )
            

    if args.run_code_3 == 'true':
         
        x_axis = [n for n in range(0, int(args.max_num_of_workers_for_code_3) + 1, 4)]
        
        for workers in x_axis:
            load_times, train_times, epoch_times = train_model(
                dataset_root_path = args.dataset_path, 
                batch_size = int(args.batch_size), 
                number_of_workers = workers,
                device = args.device, 
                optimizer = args.optimizer, 
                epochs = int(args.epochs),
                print_epochs = True if args.print_epochs == 'true' else False, 
                download_data = True if args.download_data == 'true' else False,
                no_batch_norm = True if args.no_batch_norm == 'true' else False
            )
            print("--------------------------------------------------------------") 
            print(f"Avg Data Load times with `{workers}` workers is `{'%.6f'%load_times}` sec and Avg Train Times is `{'%.6f'%train_times}` sec")
            print(f"And Avg epoch time is `{'%.4f'%epoch_times}` sec")
    


