import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_hidden = np.random.randn(hidden_size, input_size)
        self.velocities_input_hidden = np.zeros((hidden_size, input_size))
        self.weights_hidden_output = np.random.randn(output_size, hidden_size)
        self.velocities_hidden_output = np.zeros((output_size, hidden_size))
        self.biases_hidden = np.random.randn(hidden_size, 1)
        self.velocities_biases_hidden = np.zeros((hidden_size, 1))
        self.biases_output = np.random.randn(output_size, 1)
        self.velocities_biases_output = np.zeros((output_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, input_data):
        hidden_layer = self.sigmoid(np.dot(self.weights_input_hidden, input_data.T) + self.biases_hidden)
        output_layer = self.sigmoid(np.dot(self.weights_hidden_output, hidden_layer) + self.biases_output)
        return hidden_layer, output_layer

    def update_input_hidden_weights(self, input_data, hidden_gradient, learning_rate, momentum):
        self.velocities_input_hidden = (momentum * self.velocities_input_hidden) + (learning_rate * np.dot(hidden_gradient, input_data) / len(input_data))
        self.weights_input_hidden += self.velocities_input_hidden
        self.velocities_biases_hidden = (momentum * self.velocities_biases_hidden) + (learning_rate * np.mean(hidden_gradient, axis=1, keepdims=True))
        self.biases_hidden += self.velocities_biases_hidden

    def update_hidden_output_weights(self, hidden_layer, output_gradient, learning_rate, momentum):
        self.velocities_hidden_output = (momentum * self.velocities_hidden_output) + (learning_rate * np.dot(output_gradient, hidden_layer.T) / len(hidden_layer))
        self.weights_hidden_output += self.velocities_hidden_output
        self.velocities_biases_output = (momentum * self.velocities_biases_output) + (learning_rate * np.mean(output_gradient, axis=1, keepdims=True))
        self.biases_output += self.velocities_biases_output

    def train(self, input_data, output_data, epochs, mse_threshold, learning_rate, momentum):
        for epoch in range(epochs):
            try:
                hidden_layer, output_layer = self.forward_propagation(input_data)
                output_error = output_data - output_layer.T
                output_gradient = output_error.T * self.sigmoid_derivative(output_layer)
                self.update_hidden_output_weights(hidden_layer, output_gradient, learning_rate, momentum)
                hidden_error = np.dot(self.weights_hidden_output.T, output_gradient)
                hidden_gradient = hidden_error * self.sigmoid_derivative(hidden_layer)
                self.update_input_hidden_weights(input_data, hidden_gradient, learning_rate, momentum)
                error = np.mean(output_error**2, axis=0)
                if epoch % 5000 == 0:
                    print(f"Epoch: {epoch + 5000}, Error: {error}")
                if np.all(error <= mse_threshold):
                    break
            except Exception as e:
                print(f"An error occurred: {e}")
                break

    def calculate_accuracy(self, true_positive, true_negative, false_positive, false_negative):
        total_predictions = true_positive + true_negative + false_positive + false_negative
        if total_predictions == 0:
            return 0.0
        accuracy = ((true_positive + true_negative) / total_predictions) * 100
        return accuracy


def read_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            label = lines[i].strip()
            x, y = map(float, lines[i + 1].split())
            i += 2
            label_data = (label, (x, y),)
            label_data += tuple(map(int, lines[i].split()))
            i += 1
            data.append(label_data)
    return data

def split_train_test(data, train_ratio=0.9):
    np.random.shuffle(data)
    total_samples = len(data)
    train_size = int(total_samples * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

def extract_input_output(data):
    input_data = np.array([item[1] for item in data])
    output_data = np.array([item[2:] for item in data])
    return input_data, output_data

def main():
    FILE_PATH = 'D:/CI/comassign1byme/cross.txt'
    INPUT_SIZE = 2
    HIDDEN_SIZE = 8
    OUTPUT_SIZE = 2
    EPOCHS = 30000
    MSE_THRESHOLD = 0.00001

    data = read_data(FILE_PATH)
    train_data, test_data = split_train_test(data)
    input_train, output_train = extract_input_output(train_data)

    learning_rates = [0.1]
    momentums = [0.01]

    for learning_rate in learning_rates:
        for momentum in momentums:
            print(f"Learning rate = {learning_rate}, Momentum = {momentum}")
            nn = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
            nn.train(input_train, output_train, EPOCHS, MSE_THRESHOLD, learning_rate, momentum)
            input_test, output_test = extract_input_output(test_data)
            actual = output_test
            _, predicted = nn.forward_propagation(input_test)
            predicted = np.transpose(predicted)
            threshold = 0.5
            predicted = (predicted[:, 1] > threshold).astype(int)
            confusion_matrix = np.zeros((2, 2), dtype=int)
            for i in range(2):
                for j in range(2):
                    confusion_matrix[i, j] = np.sum((actual[:, i] == 1) & (predicted == j))
            true_positive, true_negative, false_positive, false_negative = (
                confusion_matrix[1, 1], confusion_matrix[0, 0], 
                confusion_matrix[0, 1], confusion_matrix[1, 0]
            )
            accuracy = nn.calculate_accuracy(true_positive, true_negative, false_positive, false_negative)

            print("\nResults:")
            print("Confusion Matrix:")
            print(confusion_matrix)
            print(f"Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main()
