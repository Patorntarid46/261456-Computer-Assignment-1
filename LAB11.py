import numpy as np

# ฟังก์ชันการเปิดใช้ Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ฟังก์ชันการหาอนุพันธ์ของ Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# อ่านข้อมูลจากไฟล์
def read_data(file_path):
    inputs = []
    outputs = []

    try:
        with open(file_path, 'r') as file:
            lines = file.read().splitlines()

            for line in lines:
                if len(line.split()) == 9:
                    data_set = line.split()
                    input_data = [int(value) for value in data_set[:8]]
                    output = int(data_set[-1])
        
                    inputs.append(input_data)
                    outputs.append(output)

    except FileNotFoundError:
        print(f'File "{file_path}" not found')
    
    return np.array(inputs), np.array(outputs)


# แยกข้อมูลเข้าและผลลัพธ์
def separate_input_output(data):
    input_data = data[:, :8]  
    output_data = data[:, 8] 
    return input_data, output_data

# ทำการ Normalize ข้อมูลให้อยู่ในช่วง 0-1 ด้วยวิธี Min-Max Normalization
def normalize_data(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data, min_vals, max_vals

# ทำการ Inverse Normalize เพื่อให้ข้อมูลกลับไปสู่ช่วงเดิม
def inverse_normalize_data(normalized_data, min_val, max_val):
    original_data = normalized_data * (max_val - min_val) + min_val
    return original_data

# ฟังก์ชันการ Forward Propagation
def forward_propagation(input_data, w_input_to_hidden, b_hidden, w_hidden_to_output, b_output):
    hidden_input = np.dot(w_input_to_hidden, input_data.T) + b_hidden
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(w_hidden_to_output, hidden_output) + b_output
    output_output = sigmoid(output_input)
    return hidden_output, output_output

# ฟังก์ชันการปรับปรุงน้ำหนักที่ Input และ Hidden Layers
def update_weights_input_hidden(input_data, hidden_output, output_output, target_output, w_hidden_to_output, w_input_to_hidden, learning_rate):
    output_error = target_output - output_output
    output_delta = output_error * sigmoid_derivative(output_output)
    hidden_error = np.dot(w_hidden_to_output.T, output_delta)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)
    w_hidden_to_output += learning_rate * np.dot(output_delta, hidden_output.T)
    w_input_to_hidden += learning_rate * np.dot(hidden_delta, input_data)

# ฟังก์ชันการปรับปรุงน้ำหนักที่ Hidden และ Output Layers
def update_weights_hidden_output(hidden_output, output_output, target_output, w_hidden_to_output, b_output, learning_rate):
    output_error = target_output - output_output
    output_delta = output_error * sigmoid_derivative(output_output)
    w_hidden_to_output += learning_rate * np.dot(output_delta, hidden_output.T)
    b_output += learning_rate * np.sum(output_delta, axis=1, keepdims=True)

def custom_train_test_split(data, target, test_size=0.1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    num_samples = len(data)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split_index = int(num_samples * (1 - test_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    return data[train_indices], data[test_indices], target[train_indices], target[test_indices]

def train_neural_network(input_data, target_output, target_epochs, mean_squared_error, learning_rate, momentum_rate):
    for epochs in range(target_epochs):
        hidden_output, output_output = forward_propagation(input_data, w_input_to_hidden, b_hidden, w_hidden_to_output, b_output)
        output_error = target_output - output_output
        output_gradient = output_error * sigmoid_derivative(output_output)
        update_weights_hidden_output(hidden_output, output_output, target_output, w_hidden_to_output, b_output, learning_rate)
        hidden_error = np.dot(w_hidden_to_output.T, output_gradient)
        hidden_gradient = hidden_error * sigmoid_derivative(hidden_output)
        update_weights_input_hidden(input_data, hidden_output, output_output, target_output, w_hidden_to_output, w_input_to_hidden, learning_rate)
        error = np.mean(output_error**2)
        
        if epochs % 10000 == 0:
            print(f"amount of epochs: {epochs}, error: {error}")
            display_results(target_output, output_output)

        if error <= mean_squared_error:
            print(f"Training completed")
            break
    predicted_output = inverse_normalize_data(output_output.T, min_val[7], max_val[7])
    
    target_output = inverse_normalize_data(target_output.T, min_val[7], max_val[7])
    
    accuracy =  np.mean(np.abs((target_output - predicted_output) / target_output) * 100)
    print(f"accuracy = {accuracy}% ")

def display_results(target_output, output_output):
    print("Actual Output    Predict Output      error")
    for i in range(min(len(target_output), len(output_output))):
        target = inverse_normalize_data(target_output[i], min_val[7], max_val[7])
        predicted = inverse_normalize_data(output_output[i], min_val[7], max_val[7])
        error = abs(predicted)
        if np.isscalar(target):
            target = np.array([target])
        if np.isscalar(predicted):
            predicted = np.array([predicted])
        if np.isscalar(error):
            error = np.array([error])
        print("    {:.2f}           {:.2f}          {:.2f}%".format(target[0], predicted[0], error[0]))

        
def train_test_split(data, target, test_size=0.1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    num_samples = len(data)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split_index = int(num_samples * (1 - test_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    return data[train_indices], data[test_indices], target[train_indices], target[test_indices]

# กำหนดที่อยู่ของไฟล์
file_path = "D:\\CI\\comassign1byme\\flooddata.txt"

#อ่านข้อมูล
input_data, output_data = read_data(file_path)

if len(input_data) > 0:
    input_data = input_data[:, :8]
    input_data, min_val, max_val = normalize_data(input_data)

    learning_rate = 0.001
    momentum_rate = 0.2
    target_output = output_data 
    target_epochs = 1000
    mean_squared_error = 0.001 

    input_s = 8
    hidden_s = 8
    output_s = 1

    w_input_to_hidden = np.random.randn(hidden_s, input_s) * np.sqrt(2 / (input_s + hidden_s))
    lastw_input_hidden = np.random.randn(hidden_s, input_s)

    w_hidden_to_output = np.random.randn(output_s, hidden_s) * np.sqrt(2 / (hidden_s + output_s))
    lastw_hidden_output = np.random.randn(output_s, hidden_s)

    b_hidden = np.random.randn(hidden_s, 1)
    lastb_hidden = np.random.randn(hidden_s, 1)

    b_output = np.random.randn(output_s, 1)
    lastb_output = np.random.randn(output_s, 1)

    # Perform 10% cross-validation
    for fold in range(10):
        X_train, X_test, y_train, y_test = train_test_split(input_data, target_output, test_size=0.1, random_state=fold)

        train_neural_network(X_train, y_train, target_epochs, mean_squared_error, learning_rate, momentum_rate)

else:
    print("No input data available")