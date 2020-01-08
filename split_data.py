import random


if __name__ == '__main__':
    data_path = './train_data/dga-feed_1.txt'
    train_data_path = './train_data/dga-feed_train.txt'
    test_data_path = './test_data/dga-feed_test.txt'



    with open(data_path, "r") as f:
        data = f.read().split('\n')

    f.close()

    random.shuffle(data)

    train_data = data[:int((len(data)+1)*.90)]
    test_data = data[int(len(data)*.90+1):]

    f_train = open(train_data_path, "a")
    f_test = open(test_data_path,"a")


    for a in train_data:
        f_train.write(a+"\n")
    for b in test_data:
        f_test.write(b+"\n")

    f_train.close()
    f_test.close()