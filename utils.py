import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import SimpleITK as sitk


class DiceLoss(torch.nn.Module):
    """A class for calculating the Dice metric loss given an image, the decoder output logits, and the target labeled image"""
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    """A function to gather different performance metrics for the segmentation tasks"""
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    """The original TransUNet test function to determine the network performance on the synapse dataset (each volume consisting of 2D image slices)"""
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+ case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list


def test_multiple_volumes(image, label, time, net, classes, patch_size=[160, 160, 160], test_save_path=None, case=None, z_spacing=1):
    """The TransVNet test function for segmented sequencing tasks"""
    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(image, time), dim=1), dim=1)
        prediction = out.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
    metric_list = np.zeros(shape=(classes-1, 2)) # np.array([[0.0, 0.0]])
    metric_i = np.array([0.0, 0.0])
    batch_size = prediction.shape[0]
    for i in range(1, classes):
        for batch_sample in range(batch_size):
            metric_i += np.array(calculate_metric_percase(prediction[batch_sample, ...] == i, label[batch_sample, ...] == i))
        # metric_list = np.append(metric_list, np.expand_dims(metric_i, axis=0), axis=0) # metric_list.append(metric_i)
        metric_list[i-1][:] = metric_i

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+ case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list


def test_multiple_volumes_generative(image_batch, label_batch, time_batch, net, name_batch, test_save_path=None):
    """The TransVNet test function for generative tasks"""
    with torch.no_grad():
        mu, log_variance, predicted_labels = net.module.encoder(image_batch, time_batch)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(mu)) # Target latent distribution
        z = p.sample()
        # Mixing the sampled tensor z(batch_size, number_of_patches, label_size) with predicted_labels(batch_size, number_of_patches, label_size) to form the input tensor of the decoder for generative purposes
        l = []
        for i in range(predicted_labels.shape[1]):
            l.append(torch.mul(z, torch.sigmoid(torch.unsqueeze(predicted_labels[:, i], -1))))
        decoder_input = torch.stack(l, 2)
        decoder_output = net.module.decoder(decoder_input)
        # # Gaussian likelihood for the reconstruction loss
        # scale = torch.exp(net.module.log_scale)
        # dist = torch.distributions.Normal(decoder_output, scale)
        # # Measure prob of seeing image under p(x|y,z)
        # log_pxz = dist.log_prob(image_batch[:,:net.module.config['n_classes'],:]) # Reconstruction loss in VAEs
        # if len(net.module.config.patches.size) == 3:
        #     log_pxz = log_pxz.mean(dim=(1, 2, 3, 4))
        # else:
        #     log_pxz = log_pxz.mean(dim=(1, 2, 3))
        # generative_output = dist.sample()
        generative_output = torch.argmax(torch.softmax(decoder_output, dim=1), dim=1) # Segmented output

    loss_mse = torch.nn.MSELoss(reduction='none')
    surrogate_model_error = torch.sum(loss_mse(predicted_labels, label_batch), dim=1)
    mu, log_variance, predicted_labels_generative = net.module.encoder(generative_output[:,0:1,:], time_batch)
    generative_error = torch.sum(loss_mse(predicted_labels_generative, label_batch), dim=1)
    # reconstruction_loss = -log_pxz
    ce_loss = torch.nn.CrossEntropyLoss()
    reconstruction_loss = ce_loss(decoder_output, image_batch)
    metric_list = torch.stack((surrogate_model_error, generative_error, reconstruction_loss), 1)

    for i in range(len(name_batch)):
        if test_save_path is not None:
            img_itk = sitk.GetImageFromArray(image_batch[i, :].cpu().detach().numpy().astype(np.float32))
            prd_itk = sitk.GetImageFromArray(generative_output[i, :].cpu().detach().numpy().astype(np.float32))
            img_itk.SetSpacing((1, 1, 1))
            prd_itk.SetSpacing((1, 1, 1))
            sitk.WriteImage(prd_itk, test_save_path + '/'+ name_batch[i] + "_pred.nii.gz")
            sitk.WriteImage(img_itk, test_save_path + '/'+ name_batch[i] + "_img.nii.gz")
    
    return (name_batch, metric_list.cpu().detach().numpy())
