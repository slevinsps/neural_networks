# -*- coding: utf-8 -*-
import numpy
import scipy.special
import matplotlib.pyplot
import scipy.misc
#%matplotlib inline

# определение класса нейронной сети
class neuralNetwork:
    # инициализировать нейронную сеть
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # задать количество узлов во входном, скрытом и выходном слое
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # коэффициент обучения
        self.lr = learningrate
        # весовые коэфициенты
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # функция активации (сигмоида)
        self.activation_function = lambda x: scipy.special.expit(x)
    
        pass

    # тренировка нейронной сети
    def train (self, inputs_list, targets_list) :
        # преобразование списка входных значений
        # в двухмерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        # рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)
        # ошибки выходного слоя =
        # (целевое значение - фактическое значение)
        output_errors = targets - final_outputs
        # ошибки скрытого слоя - это ошибки output_errors,
        # распределенные пропорционально весовым коэффициентам связей
        # и рекомбинированные на скрытых узлах
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # обновить весовые коэффициенты для связей между
        # скрытым и выходным слоями
        self.who += self.lr * numpy .dot ((output_errors *
                                           final_outputs * (1.0 - final_outputs)), numpy.transpose (hidden_outputs))
        # обновить весовые коэффициенты для связей между
        # входным и скрытым слоями
        self.wih += self.lr * numpy.dot((hidden_errors *
                                         hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass

    # опрос нейронной сети
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        #print (inputs)
        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        # рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
    
def test(new_neural, data_file):
    error_arr = []
    training_data_list = data_file.readlines()
    
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        correct = int(all_values[0])
        answer_arr = new_neural.query(inputs)
        label = numpy.argmax(answer_arr)
        #print("Ответ сети =", label, "; Верный ответ =", correct, "\n")
        
        if (label == correct):
            error_arr.append(1)
        else:
            error_arr.append(0)
            
    #print(error_arr)
    scorecard_array = numpy.asarray(error_arr)
    print("Эффективность =", scorecard_array.sum()/scorecard_array.size)
    
def start_train(new_neural, data_file):
    training_data_list = data_file.readlines()
    #print(data_list[0])
    # перебрать все записи в тренировочном наборе данных
    for record in training_data_list:
        # получить список значений, используя символы запятой (1,1)
        # в качестве разделителей
        all_values = record.split(',')
        # масштабировать и сместить входные значения
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # создать целевые выходные значения (все равны 0,01, за исключением
        # желаемого маркерного значения, равного 0,99)
        targets = numpy.zeros(outputnodes) + 0.01
        # all_values[0] - целевое маркерное значение для данной записи
        targets[int(all_values[0])] = 0.99
        new_neural.train(inputs, targets)
        pass

def load_neural_data(inputnodes, hiddennodes, outputnodes):
    wih = numpy.zeros(shape=(hiddennodes, inputnodes))
    who = numpy.zeros(shape=(outputnodes, hiddennodes))
    data_write_file = open("mnist/wih.txt", "r")
    #print(new_neural.wih.shape)
    all_lines = data_write_file.readlines()
    for i in range(hiddennodes):
        line = all_lines[i].split(' ')
        for j in range(inputnodes):
            wih[i][j] = float(line[j])
    data_write_file.close()
    
    data_write_file = open("mnist/who.txt", "r")
    #print(new_neural.wih.shape)
    all_lines = data_write_file.readlines()
    for i in range(outputnodes):
        line = all_lines[i].split(' ')
        for j in range(hiddennodes):
            who[i][j] = float(line[j])
    data_write_file.close()
    return wih, who

def write_neural_data(wih, who):  

    data_write_file = open("mnist/wih.txt", "w")
    #print(new_neural.wih.shape)
    for row in wih:
        for col in row:
            data_write_file.write(str(col) + " ")
        data_write_file.write("\n")
    data_write_file.close()
    
    data_write_file = open("mnist/who.txt", "w")
    #print(new_neural.wih.shape)
    for row in who:
        for col in row:
            data_write_file.write(str(col) + " ")
        data_write_file.write("\n")
    data_write_file.close()

inputnodes = 784
hiddennodes = 200 # поднялось до 0.95
outputnodes = 10
learningrate = 0.2 # Дает 0.9513 при 0.3 - 0.9494
new_neural = neuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate)



new_neural.wih, new_neural.who = load_neural_data(inputnodes, hiddennodes, outputnodes)
'''
data_file = open("mnist/mnist_train.csv", "r")
start_train(new_neural, data_file)
data_file.close()
'''
#write_neural_data(new_neural.wih, new_neural.who)

'''
test_file = open("mnist/mnist_mytest.csv", "r")
test(new_neural, test_file)
test_file.close()
'''
image_file_name = "two.png"
img_array = scipy.misc.imread(image_file_name, flatten=True)
img_data = 255.0 - img_array.reshape(784) 
print(img_data.size)
img_data = (img_data / 255.0 * 0.99) + 0.01
print(img_data)
ans = new_neural.query(img_data)
print(ans)
label = numpy.argmax(ans)
print("Ответ = ", label)
#matplotlib.pyplot.imshow(img_data, cmap='Greys', interpolation='None')

