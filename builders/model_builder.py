from model.tapnet import Network


def build_model(model_name, num_classes):
    if model_name == 'tapnet':
        return Network(num_classes=num_classes)

