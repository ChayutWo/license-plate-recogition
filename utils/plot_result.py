import pandas as pd
import matplotlib.pyplot as plt

def plot_result(train_loss, test_loss, num_epochs):
    plt.rcParams['figure.figsize'] = [10,6]
    plt.rcParams.update({'font.size':14})
    result_df = pd.DataFrame({'iteration':range(1, num_epochs+1) ,'train_loss':train_loss, 'test_loss':test_loss})
    result_df.to_csv('result.csv', index = False)
    fig = plt.figure()
    plt.plot(train_loss, label = 'train_loss')
    plt.plot(test_loss, label = 'test_loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.ylim(bottom = 0)
    plt.title('Learning progression')
    plt.legend()
    fig.savefig('learning_progression.png')
    return
