
from NNet import *

if __name__ == "__main__":
    net = Network([3,10,1])
    data = np.array([])

    f = open("Cu_conc.txt", 'r')
    data = f.readlines()
    f.close()


    max_values = [2884.99481865 , 360.33160622, 2832.02140022 ,  24.72      ]
    min_values = [196.46820798 , 30.36791137, 866.27118977 , 11.95      ]

    learn_data = []
    for line in data:
        conc = float(line.split()[0])
        spt_name = line.split()[1].strip()
        train_data = GetIntensities(spt_name)
        train_data = np.append(train_data, conc)

        inputs = train_data[0:-1].reshape(3, 1)
        inputs[0] = (inputs[0] - min_values[0]) / (max_values[0] - min_values[0])
        inputs[1] = (inputs[1] - min_values[1]) / (max_values[1] - min_values[2])
        inputs[2] = (inputs[2] - min_values[2]) / (max_values[2] - min_values[2])

        outputs = train_data[-1].reshape(1, 1)
        outputs[0] = (outputs[0] - min_values[3]) / (max_values[3] - min_values[3])

        xj = (inputs, outputs)
        learn_data.append(xj)

    print(len(learn_data))

    net.SGD(learn_data, 2500, len(learn_data), 2)


    delta_array = np.array([])
    for entry in learn_data:
        received_value = net.feedforward(entry[0])
        real_value_received = received_value*(24.72 - 11.95) + 11.95

        real_value_expected = entry[1] * (24.72 - 11.95) + 11.95
        print(" Expected: ", real_value_expected , " Received : ", real_value_received, " Delta : ", abs(real_value_expected - real_value_received))

        delta_array = np.append(delta_array, abs(real_value_expected - real_value_received))



    plt.plot(np.arange(0, len(delta_array),1), delta_array, '--ro')
    plt.show()




