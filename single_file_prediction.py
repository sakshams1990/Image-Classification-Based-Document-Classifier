import os

import PIL.Image
import numpy
from pdf2image import convert_from_path
from get_values_from_config_file import image_extension, device, poppler_path
from dataset_loader import val_test_image_transforms
import torch
import torch.nn.functional as F
from statistical_analysis import perform_statistical_analysis
from dataset_loader import get_datasets_dataloaders_for_model


def predict(model, test_file_path):
    predicted_result = {}
    train_dataset, train_dataload, val_dataset, val_dataload, \
        test_dataset, test_dataload = get_datasets_dataloaders_for_model()
    index_to_class, num_classes, category_names = perform_statistical_analysis(train_dataset, val_dataset, test_dataset)

    transform = val_test_image_transforms()
    file_extension = os.path.splitext(test_file_path)[1]
    img = None
    if file_extension == '.pdf':
        img = convert_from_path(test_file_path, poppler_path=poppler_path)
        img = img[0]
    elif file_extension in image_extension:
        img = test_file_path
        img = PIL.Image.open(img).convert('RGB')
    else:
        print('Invalid File!!')
    test_img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_img_tensor)
        ps = torch.exp(out)

        topk, topclass = ps.topk(1, dim=1)
        class_name = index_to_class[topclass.cpu().numpy()[0][0]]

        pred = model(test_img_tensor.to(device))
        pred = F.softmax(pred, dim=1)
        pred = pred.detach().cpu().numpy().flatten()
        score = round(numpy.amax(pred), 3)

        predicted_result['predictedClass'] = class_name
        predicted_result['confidenceScore'] = score
        return predicted_result


if __name__ == '__main__':
    base_model = torch.load('Best Model/best_model_state.pt')
    test_file = "Dataset/Output/test/INVOICE/ETC DOC.jpg"
    pred_res = predict(base_model, test_file)
    print(pred_res)
