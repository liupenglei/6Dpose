import os
import time
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import scipy
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import scipy.misc
from darknet import Darknet
import dataset
from utils import *
from MeshPly import MeshPly

# Create new directory
def makedirs(path):
    if not os.path.exists( path ):
        os.makedirs( path )

def valid(datacfg, modelcfg, weightfile):
    def truths_length(truths, max_num_gt=50):
        for i in range(max_num_gt):
            if truths[i][1] == 0:
                return i

    # Parse configuration files
    data_options = read_data_cfg(datacfg)
    valid_images = data_options['valid']
    meshname     = data_options['mesh']
    backupdir    = data_options['backup']
    name         = data_options['name']
    gpus         = data_options['gpus'] 
    fx           = float(data_options['fx'])
    fy           = float(data_options['fy'])
    u0           = float(data_options['u0'])
    v0           = float(data_options['v0'])
    im_width     = int(data_options['width'])
    im_height    = int(data_options['height'])
    if not os.path.exists(backupdir):
        makedirs(backupdir)

    # Parameters
    seed = int(time.time())
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)
    save            = True
    visualize       = False
    testtime        = True
    num_classes     = 1
    testing_samples = 0.0
    edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    

    # Read object model information, get 3D bounding box corners
    mesh      = MeshPly("LINEMOD/duck/duck.ply")
    vertices  = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    corners3D = get_3D_corners(vertices)
    try:
        diam  = float(options['diam'])
    except:
        diam  = calc_pts_diameter(np.array(mesh.vertices))
        
    # Read intrinsic camera parameters
    intrinsic_calibration = get_camera_intrinsic(u0, v0, fx, fy)

    # Get validation file names
    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]
    
    # Specicy model, load pretrained weights, pass to GPU and set the module in evaluation mode
    model = Darknet(modelcfg)
    model.print_network()
    model.load_weights(weightfile)
    model.cuda()
    model.eval()
    test_width    = model.test_width
    test_height   = model.test_height
    num_keypoints = model.num_keypoints 
    num_labels    = num_keypoints * 2 + 3

    # Get the parser for the test dataset
    valid_dataset = dataset.listDataset(valid_images, 
                                        shape=(test_width, test_height),
                                        shuffle=False,
                                        transform=transforms.Compose([transforms.ToTensor(),]))

    # Specify the number of workers for multiple processing, get the dataloader for the test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, **kwargs) 

    logging("   Testing {}...".format(name))
    logging("   Number of test samples: %d" % len(test_loader.dataset))
    # Iterate through test batches (Batch size for test data is 1)
    for batch_idx, (data, target) in enumerate(test_loader):

        img_test = scipy.misc.imread('./LINEMOD/duck/JPEGImages/000000.jpg')
        img_tensor = torch.tensor(img_test)
        img_tensor = torch.unsqueeze(img_tensor,0)
        img_tensor = torch.transpose(img_tensor, 1, 3)
        img_tensor = torch.transpose(img_tensor, 2, 3)
        img_tensor = img_tensor.float()
        
        # Images
        img = data[0, :, :, :]
        
        img = img.numpy().squeeze()
        img = np.transpose(img, (1, 2, 0))
        
        t1 = time.time()
        # Pass data to GPU
        data = img_tensor.cuda()
        target = target.cuda()
        # Wrap tensors in Variable class, set volatile=True for inference mode and to use minimal memory during inference
        data = Variable(data, volatile=True)
        t2 = time.time()
        # Forward pass
        output = model(data).data  
        t3 = time.time()
        # Using confidence threshold, eliminate low-confidence predictions
        all_boxes = get_region_boxes(output, num_classes, num_keypoints)        
        t4 = time.time()
        # Evaluation
        # Iterate through all batch elements
        for box_pr, target in zip([all_boxes], [target[0]]):
            # For each image, get all the targets (for multiple object pose estimation, there might be more than 1 target per image)
            truths = target.view(-1, num_keypoints*2+3)
            # Get how many objects are present in the scene
            num_gts    = truths_length(truths)
            # Iterate through each ground-truth object
            for k in range(num_gts):
                box_gt = list()
                for j in range(1, 2*num_keypoints+1):
                    box_gt.append(truths[k][j].cpu())
                box_gt.extend([1.0, 1.0])
                box_gt.append(truths[k][0].cpu())

                # Denormalize the corner predictions 
         
                corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')     
                corners2D_pr[:, 0] = corners2D_pr[:, 0] * im_width
                corners2D_pr[:, 1] = corners2D_pr[:, 1] * im_height
    
                # Compute [R|t] by pnp
                R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_pr, np.array(intrinsic_calibration, dtype='float32'))
                
                return R_pr
 


def test(modelcfg, weightfile, img):
    
    preprocess_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Darknet(modelcfg)
    test_width    = model.test_width
    test_height   = model.test_height
    num_keypoints = model.num_keypoints 
    num_labels    = num_keypoints * 2 + 3
    num_classes = 1

    mesh      = MeshPly('./LINEMOD/duck/duck.ply')
    vertices  = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    corners3D = get_3D_corners(vertices)

    
    u0 = 598.7
    v0 = 354.8
    fx = 939.434
    fy = 942.118
    
    intrinsic_calibration = get_camera_intrinsic(u0, v0, fx, fy)

    # model.print_network()
    model.load_weights(weightfile)
    model.cuda()
    model.eval()
    image_tensor = preprocess_transform(img)
   

    image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.to(device)
    output = model(image_tensor)
    all_boxes = get_region_boxes(output, num_classes, num_keypoints)   

    for i in range(len(all_boxes)):
        all_boxes[i] = all_boxes[i].detach().numpy()
        
    box_pr = np.array(np.reshape(all_boxes[:18], [9, 2]), dtype='float32')

    box_pr[:, 0] = box_pr[:, 0] * 640
    box_pr[:, 1] = box_pr[:, 1] * 480

    R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'),  box_pr, np.array(intrinsic_calibration, dtype='float32'))
                
    print("juzhen:",R_pr)
    
  

    aa1 = (box_pr[1,0] +  box_pr[2,0] + box_pr[3,0] + box_pr[4,0])/4
    bb1 = (box_pr[1,1] +  box_pr[2,1] + box_pr[3,1] + box_pr[4,1])/4
    
    confi = all_boxes[18]
   
    print("confidence : ", all_boxes[18])

    if all_boxes[18] < 0.03:

        box_pr[0, 0] = 0
        box_pr[0, 1] = 0
    
    else:
        box_pr[0, 0] = box_pr[0, 0] 
        box_pr[0, 1] = box_pr[0, 1] 

    
    return int(box_pr[0,0]), int(box_pr[0,1]) ,confi ,R_pr




def test1(modelcfg, weightfile, img):
    
    preprocess_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Darknet(modelcfg)
    test_width    = model.test_width
    test_height   = model.test_height
    num_keypoints = model.num_keypoints 
    num_labels    = num_keypoints * 2 + 3
    num_classes = 1
    # model.print_network()
    model.load_weights(weightfile)
    model.cuda()
    model.eval()
    image_tensor = preprocess_transform(img)
   

    image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.to(device)
    output = model(image_tensor)
    all_boxes = get_region_boxes(output, num_classes, num_keypoints)   

    for i in range(len(all_boxes)):
        all_boxes[i] = all_boxes[i].detach().numpy()
        
    box_pr = np.array(np.reshape(all_boxes[:18], [9, 2]), dtype='float32')

    print("confidence : ", all_boxes[18])
    if all_boxes[18] < 0.01:

        box_pr[0, 0] = 0
        box_pr[0, 1] = 0
    
    else:
        box_pr[0, 0] = box_pr[0, 0] * 640
        box_pr[0, 1] = box_pr[0, 1] * 480

        a = (box_pr[1, 0] + box_pr[2, 0] + box_pr[3, 0] + box_pr[4, 0])/4
        b = (box_pr[1, 1] + box_pr[2, 1] + box_pr[3, 1] + box_pr[4, 1])/4
        a = a * 640
        b = b * 480

    return int(a), int(b)



#datacfg    = 'cfg/duck.data'

modelcfg   = 'cfg/MFPN-yolov4-pose.cfg'

weightfile = 'backup/duck/MFPN.weights'

img_test = Image.open('./save.png')

x,y,confi,rot= test(modelcfg, weightfile, img_test)

#x1,y1 = test1(modelcfg, weightfile, img_test)

print("x = ", x , ",y = ",y ,"confidence = ",confi)
