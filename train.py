from model_poly2mat import poly2mat

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
import pickle, torch, cv2, os


def EulerTransformation(polygon, angle, center=(0, 0)):
    angle = np.radians(angle)
    cos = np.cos(angle)
    sin = np.sin(angle)
    cx, cy = center
    polygonRot = []
    for x, y in polygon:
        tx, ty = x - cx, y - cy
        rx = tx * cos - ty * sin
        ry = tx * sin + ty * cos
        x = rx + cx
        y = ry + cy
        polygonRot.append((x, y))
    return np.array(polygonRot)

def MakeCoordinatesNMat(rangeAngle=[-45, 45], rangeNum=[1, 4], sizeWH=[128, 64], wRange=[8, 128], hRange=[8, 128], blur=5, scale=5, dilate=3):
    xRect = np.random.uniform(sizeWH[0])
    yRect = np.random.uniform(sizeWH[1])
    wRect = np.random.uniform(*wRange)
    hRect = np.random.uniform(*hRange)
    xSta = xRect-wRect/2
    ySta = yRect-hRect/2
    xEnd = xRect+wRect/2
    yEnd = yRect+hRect/2
    label = np.zeros([int(sizeWH[1]*scale*1.5), int(sizeWH[0]*scale*1.5)], dtype=np.uint8)
    num = np.random.randint(rangeNum[0]-1, rangeNum[1])
    coordinates = [np.array([[xSta, ySta], [xSta, yEnd], [xEnd, yEnd], [xEnd, ySta]])]
    for i in range(num):
        xStaNega, xEndNega = np.sort(np.random.uniform(xSta-wRect/2, xEnd+wRect/2, 2))
        yStaNega, yEndNega = np.sort(np.random.uniform(ySta-hRect/2, yEnd+hRect/2, 2))
        coordinates.append(np.array([[xStaNega, yStaNega], [xStaNega, yEndNega], [xEndNega, yEndNega], [xEndNega, yStaNega]]))
    cv2.fillPoly(label, [(coordinate*scale+np.array([sizeWH[0], sizeWH[1]])*scale*0.2).astype(int) for coordinate in coordinates], color=255)
    contours, _ = cv2.findContours(label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _h, _w = label.shape
    angle = np.random.uniform(*rangeAngle)
    contours = [EulerTransformation(contour[:,0], -angle, (_w/2, _h/2)).astype(int) for contour in contours]
    label = cv2.fillPoly(np.zeros([int(sizeWH[1]*scale*1.5), int(sizeWH[0]*scale*1.5)], dtype=np.uint8), contours, 255)[int(sizeWH[1]*scale*0.25):-int(sizeWH[1]*scale*0.25),int(sizeWH[0]*scale*0.25):-int(sizeWH[0]*scale*0.25)]
    contours = [contour/scale-np.array([sizeWH[0], sizeWH[1]])*0.25 for contour in contours]
    data = label
    if dilate > 0:
        label = cv2.dilate(label, np.ones([dilate*scale, dilate*scale]))
    if blur > 0:
        label = cv2.blur(label, ksize=(blur*scale, blur*scale))
    data = cv2.resize(data, dsize=(sizeWH[0], sizeWH[1]), interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, dsize=(sizeWH[0], sizeWH[1]), interpolation=cv2.INTER_LINEAR)
    return data, label, contours

def collate_fn_padd(batch):
    dataList = []
    labelList = []
    _coordinatesList = []
    for pack in batch:
        data, label, coordinates = pack
        dataList.append(data)
        labelList.append(label)
        _coordinatesList.append(coordinates)
    maxVertex = max([coordinates.shape[0] for coordinates in _coordinatesList])
    coordinatesList = torch.zeros(len(dataList), maxVertex, 6)
    coordinatePadsList = torch.zeros(len(dataList), maxVertex).to(torch.bool)
    for idx, coordinates in enumerate(_coordinatesList):
        coordinatesList[idx,:len(coordinates)] = coordinates
        coordinatePadsList[idx,:len(coordinates)] = True
    return torch.stack(dataList), torch.stack(labelList), coordinatesList, coordinatePadsList

class PolyDataset(Dataset):
    def __init__(self, num, tfData, tfLabel, rangeAngle=[-45, 45], rangeNum=[3, 8], sizeWH=[128, 128], wRange=[-16, 144], hRange=[-16, 144], scale=2, blur=5, dilate=7):
        self.num = num
        self.tfData = tfData
        self.tfLabel = tfLabel
        self.rangeAngle = rangeAngle
        self.sizeWH = sizeWH
        self.rangeNum=rangeNum
        self.hRange = hRange
        self.wRange = wRange
        self.scale=scale
        self.dilate = dilate
        self.blur=blur

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        data, label, coordinates = MakeCoordinatesNMat(rangeAngle=self.rangeAngle, rangeNum=self.rangeNum, sizeWH=self.sizeWH, wRange=self.wRange, hRange=self.hRange, scale=self.scale, blur=self.blur, dilate=self.dilate)
        data = self.tfData(data)
        coordinates = np.concatenate([np.concatenate([coordinate, np.roll(coordinate, shift=-1), np.roll(coordinate, shift=1)], 1) for coordinate in coordinates], 0)
        coordinates = torch.Tensor(coordinates)/torch.Tensor([*self.sizeWH]*3)
        label = self.tfLabel(label)
        return data, label, coordinates

class IoULoss(torch.nn.Module):
    def __init__(self, targetLabel, smooth, weight):
        super(IoULoss, self).__init__()
        self.targetLabel = targetLabel
        self.smooth = smooth
        self.weight = weight
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, logit, target):
        num, c, w, h = logit.shape
        logit = self.sigmoid((logit[:,self.targetLabel]-0.5)*20)
        target = target[:,self.targetLabel]
        intersection = target * logit
        union = target + logit - intersection
        intersection = intersection.view(num, -1).sum(1) + self.smooth
        union = union.view(num, -1).sum(1) + self.smooth
        iou = intersection / union
        return (1 - iou).mean() * self.weight

if __name__ == "__main__":
    tfLabel = transforms.Compose([
        transforms.ToTensor()
    ])
    tfData = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    pathModel = "model.pickle"
    try:
        ckp = pickle.load(open("model.pickle", "rb"))
        modelOptions = ckp["model_options"]
        model = poly2mat(**modelOptions)
        model.load_state_dict(ckp["model_state_dict"])
        print("model loaded")
    except:
        print("initial")
        modelOptions = {
            "chanIn" : 1,
            "chanPoly" : 64,
            "chanOut" : 1,
            "scaleList" : [1.0, 0.7, 0.4, 0.1],
            "chanMidList" : [64, 64, 64, 64],
            "repeatAttList"  :[1, 1, 1, 1]
        }
        model = poly2mat(**modelOptions)
    device = torch.device("cuda:0")
    model.to(device)
    lossFuncMSE = torch.nn.MSELoss()
    lossFuncIoU = IoULoss(0, 4, 1.0)
    learningRate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    numCurrnent = 0

    rangeAngle=[-45, 45]
    rangeNum=[3, 5]
    sizeWH=[32, 32]
    wRange=[sizeWH[0]//8, sizeWH[0]+sizeWH[0]//8]
    hRange=[sizeWH[1]//8, sizeWH[1]+sizeWH[1]//8]
    scale=10
    blur=1
    dilate=0
    batchsize = 64
    for c in range(16):
        sizeWH[0] += 2
        sizeWH[1] += 2
        wRange=[sizeWH[0]//8, sizeWH[0]+sizeWH[0]//8]
        hRange=[sizeWH[1]//8, sizeWH[1]+sizeWH[1]//8]
        model.train()
        dataloader = DataLoader(PolyDataset(200000, tfData, tfLabel, rangeAngle=rangeAngle, rangeNum=rangeNum, sizeWH=sizeWH, wRange=wRange, hRange=hRange, scale=scale, blur=blur, dilate=dilate), batch_size=batchsize, shuffle=True, num_workers=12, collate_fn=collate_fn_padd)
        for pack in dataloader:
            datas, labels, coordinates, coordinatePads = pack
            # datas = datas.to(device)
            datas = torch.zeros(*datas.shape, device=device)
            coordinates = coordinates.to(device)
            coordinatePads = coordinatePads.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(datas, coordinates, coordinatePads)

            loss = lossFuncMSE(outputs, labels)*0.5
            maskEdge = tuple([(labels>0)*(labels<1)])
            loss += lossFuncMSE(outputs[maskEdge], labels[maskEdge])
            loss += lossFuncIoU(outputs, labels) * 0.3
            loss.backward()

            optimizer.step()

            numCurrnent += len(datas)
            print(c+1, numCurrnent, loss.item(), end="\r")
        print()
        model.eval()
        with torch.no_grad():
            data, label, coordinatesRaw = MakeCoordinatesNMat(rangeAngle=rangeAngle, rangeNum=rangeNum, sizeWH=sizeWH, wRange=wRange, hRange=hRange, scale=scale, blur=blur, dilate=dilate)
            datas = tfData(data).unsqueeze(0).to(device)
            datas = torch.zeros(*datas.shape, device=device)
            coordinates = np.concatenate([np.concatenate([coordinate, np.roll(coordinate, shift=-1), np.roll(coordinate, shift=1)], 1) for coordinate in coordinatesRaw], 0)
            coordinates = torch.Tensor(coordinates)/torch.Tensor([*sizeWH]*3)
            coordinates = coordinates.unsqueeze(0).to(device)
            coordinatePads = torch.ones(*coordinates[:,:,0].shape, dtype=torch.bool).to(device)
            labels = tfLabel(label).unsqueeze(0).to(device)
            coordinatesRaw = [np.array(coordinate.tolist() + coordinate[:1].tolist()) for coordinate in coordinatesRaw]
            for isPoly in [True]:
                outputs = model(datas, coordinates, coordinatePads) if isPoly else model(datas)
                plt.figure(figsize=(10, 2.4))
                plt.suptitle("Epoch %d"%(c+1))
                plt.subplot(1,6,1)
                plt.title("input")
                plt.imshow(datas[0][0].detach().cpu().numpy(), vmin=-1, vmax=1, extent=[0, sizeWH[0], 0, sizeWH[1]], origin="lower")
                if isPoly:
                    for coordinate in coordinatesRaw:
                        plt.plot(coordinate[:,0], coordinate[:,1], c="red", alpha=0.5, linewidth=1)
                plt.xlim(0, sizeWH[0])
                plt.ylim(0, sizeWH[1])
                # plt.axis("off")
                plt.subplot(1,6,2)
                plt.title("AI")
                plt.imshow(outputs[0][0].detach().cpu().numpy(), vmin=0, vmax=1, extent=[0, sizeWH[0], 0, sizeWH[1]], origin="lower")
                plt.xlim(0, sizeWH[0])
                plt.ylim(0, sizeWH[1])
                plt.axis("off")
                plt.subplot(1,6,3)
                plt.title("GT")
                plt.imshow(labels[0][0].detach().cpu().numpy(), vmin=0, vmax=1, extent=[0, sizeWH[0], 0, sizeWH[1]], origin="lower")
                plt.xlim(0, sizeWH[0])
                plt.ylim(0, sizeWH[1])
                plt.axis("off")
                plt.subplot(1,6,4)
                plt.title("ABS diff")
                plt.imshow(abs(outputs[0][0].detach().cpu().numpy() - labels[0][0].detach().cpu().numpy()), vmin=0, vmax=1, cmap="rainbow", extent=[0, sizeWH[0], 0, sizeWH[1]], origin="lower")
                plt.xlim(0, sizeWH[0])
                plt.ylim(0, sizeWH[1])
                plt.axis("off")
                plt.subplot(1,6,5)
                plt.title("AI+polygon")
                plt.imshow(outputs[0][0].detach().cpu().numpy(), vmin=0, vmax=1, extent=[0, sizeWH[0], 0, sizeWH[1]], origin="lower")
                for coordinate in coordinatesRaw:
                    plt.plot(coordinate[:,0], coordinate[:,1], c="red", alpha=0.5, linewidth=1)
                plt.xlim(0, sizeWH[0])
                plt.ylim(0, sizeWH[1])
                plt.axis("off")
                plt.subplot(1,6,6)
                plt.title("GT+polygon")
                plt.imshow(labels[0][0].detach().cpu().numpy(), vmin=0, vmax=1, extent=[0, sizeWH[0], 0, sizeWH[1]], origin="lower")
                for coordinate in coordinatesRaw:
                    plt.plot(coordinate[:,0], coordinate[:,1], c="red", alpha=0.5, linewidth=1)
                plt.xlim(0, sizeWH[0])
                plt.ylim(0, sizeWH[1])
                plt.axis("off")
                plt.tight_layout()
                os.makedirs("results", exist_ok=True)
                plt.savefig("results/%.4d.jpg"%(c+1))
                plt.close()
    pickle.dump(
        {
            "model_options" : modelOptions,
            "model_state_dict" : {k : v.cpu() for k, v in model.state_dict().items()}
        },
        open("model.pickle", "wb")
    )