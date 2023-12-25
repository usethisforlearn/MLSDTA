from torch_geometric.data import DataLoader
from utils import *
from create_data import create_dataset
import time


from models.MLSDTA import MLSDTA

datasets = ['davis','kiba']

modelings = [MLSDTA]
cuda_name = "cuda:0"
print('cuda_name:', cuda_name)

TEST_BATCH_SIZE = 512

result = []
for dataset in datasets:

    start_time = time.time()

    test_data_file = "./data/" + dataset + "_testdata_"+str(TEST_BATCH_SIZE)+".data"
    if not os.path.isfile(test_data_file):
        _, test_data = create_dataset(dataset)
        torch.save(test_data, test_data_file)
    else:
        test_data = torch.load(test_data_file)


    print('完成dataset加载。')
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)

    print('完成dataloader加载。')
    end_time = time.time()
    all_time = end_time - start_time
    print('数据准备一共耗时:', all_time, '秒')

    for modeling in modelings:
        model_st = modeling.__name__
        print('\npredicting for ', dataset, ' using ', model_st)
        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)

        model_file_name = 'model_' + model_st + '_' + dataset + '.model'
        if os.path.isfile(model_file_name):
            # model.load_state_dict(torch.load(model_file_name, map_location={'cuda:2':'cuda:0', 'cuda:1':'cuda:0'}), strict=False)
            model.load_state_dict(torch.load(model_file_name, map_location=torch.device('cpu')), strict=False)
            
            G,P = predicting(model, device, test_loader)
            ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P),rm2(G,P)]
            ret =[dataset, model_st] + [round(e, 3) for e in ret]
            result += [ ret ]
            print('dataset,model,rmse,mse,pearson,spearman,ci,rm2')
            print(ret)
        else:
            print('model is not available!')
with open('predict_result.csv', 'w') as f:
    f.write('dataset,model,rmse,mse,pearson,spearman,ci,rm2\n')
    for ret in result:
        f.write(','.join(map(str,ret)) + '\n')
