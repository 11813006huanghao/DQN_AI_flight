import pdb
import sys

import pyautogui as pg
import cv2
import os
import random
import numpy as np
from collections import deque
import torch
from torch.autograd import Variable
import torch.nn as nn
import time
import win32api
import win32con
from pymouse import PyMouse
import requests
import base64

ACTIONS = 10 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 33. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.7 # final value of epsilon
INITIAL_EPSILON = 0.7 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH_SIZE = 1 # size of minibatch
FRAME_PER_ACTION = 1
UPDATE_TIME = 5
GAME_STATE=0   #0表示游戏未开始   1表示第一条命正在游戏中   2表示第一条命死亡后   3表示第二条命正在游戏中   4表示第二条命死亡，游戏结束
accessToken="24.f4ce1204519e503b16469d588bab6d3b.2592000.1625749220.282335-23021910"
red_damage=0
yellow_damage=0
orange_damage=0
hight=1500
enemyInRange=0
areaRatio=0

class DeepNetWork(nn.Module):
    def __init__(self,):
        super(DeepNetWork,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=100, stride=20,padding=50),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1600,256),
            nn.ReLU()
        )
        self.out = nn.Linear(256,11)

    def forward(self, x):
        x = self.conv1(x);
        #print(x.detach().numpy().shape)
        x = self.conv2(x);
        #print(x.detach().numpy().shape)
        x = self.conv3(x);
        #print(x)
        #print(x.detach().numpy().shape)
        x = x.view(x.size(0),-1)
        #print(x.detach().numpy().shape)
        x = self.fc1(x);
        return self.out(x)

class BrainDQNMain(object):
    def save(self):
        print("save model")
        self.saveCount+=1
        #self.timeStep=0;
        torch.save(self.Q_net.state_dict(), 'paramsFlight.pth')

    def load(self):
        if os.path.exists("paramsFlight.pth"):
            print("load model param")
            self.Q_net.load_state_dict(torch.load('paramsFlight.pth'))
            self.Q_netT.load_state_dict(torch.load('paramsFlight.pth'))

    def __init__(self,actions):
        self.replayMemory = deque() # init some parameters   deque返回一个类似双向链表的数据结构，即可以从两端操作的数据结构
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        self.Q_net=DeepNetWork()
        self.Q_netT=DeepNetWork();
        self.load()
        self.loss_func=nn.MSELoss()
        self.saveCount=0
        self.randomAction=[0,2,4,6,8,10]
        self.actionIndex=0
        self.randomActionIndex=0;
        LR=1e-6
        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=LR)

    def getRondomAction(self):
        actionIndex=self.randomAction[self.randomActionIndex]
        self.randomActionIndex=(self.randomActionIndex+1)%len(self.randomAction)
        return actionIndex

    def train(self): # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]  # 32*4*80*80
        action_batch = [data[1] for data in minibatch]  # 32*2
        reward_batch = [data[2] for data in minibatch]  # 32*1
        nextState_batch = [data[3] for data in minibatch]  # Step 2: calculate y
        # print("action_batch", action_batch)
        # print("reward_batch",reward_batch)
        # print("nextState_batch",nextState_batch)
        y_batch = np.zeros([BATCH_SIZE, 1])
        nextState_batch = np.array(nextState_batch)  # print("train next state shape")
        # print(nextState_batch.shape)
        nextState_batch = torch.Tensor(nextState_batch)
        action_batch = np.array(action_batch)

        index = action_batch.argmax(axis=1)  # 32*1  即一维数组，共32个元素

        # print("action "+str(index))
        index = np.reshape(index, [BATCH_SIZE, 1])
        action_batch_tensor = torch.LongTensor(index)

        QValue_batch = self.Q_netT(nextState_batch)

        QValue_batch = QValue_batch.detach().numpy()
        print("Qvalue_batch: ", QValue_batch)
        print("reward_batch: ", reward_batch)
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch[i][0] = reward_batch[i]
            else:
                # 这里的QValue_batch[i]为数组，大小为所有动作集合大小，QValue_batch[i],代表
                # 做所有动作的Q值数组，y计算为如果游戏停止，y=rewaerd[i],如果没停止，则y=reward[i]+gamma*np.max(Qvalue[i])
                # 代表当前y值为当前reward+未来预期最大值*gamma(gamma:经验系数)
                y_batch[i][0] = reward_batch[i] + GAMMA * np.max(QValue_batch[i])

        y_batch = np.array(y_batch)
        y_batch = np.reshape(y_batch, [BATCH_SIZE, 1])
        state_batch_tensor = Variable(torch.Tensor(state_batch))
        y_batch_tensor = Variable(torch.Tensor(y_batch))
        y_predict = self.Q_net(state_batch_tensor).gather(1, action_batch_tensor)
        loss = self.loss_func(y_predict, y_batch_tensor)
        print("loss is " + str(loss))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.timeStep % UPDATE_TIME == 0:
            self.Q_netT.load_state_dict(self.Q_net.state_dict())
            self.save()


    def storeToMemory(self,action,nextObservation,reward,terminal): #print(nextObservation.shape)
        #newState = np.append(self.currentState[3:,:,:],nextObservation,axis = 0) # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        newState=np.append(self.currentState[3:,:,:],nextObservation,axis=0)
        newState=np.array(newState)
        #print(newState.shape)
        self.replayMemory.append((self.currentState,action,reward,newState,terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE: # Train the network
            self.train()

        # print info
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        #print ("TIMESTEP", self.timeStep, "/ STATE", state, "/ EPSILON", self.epsilon)
        self.currentState = newState
        self.timeStep += 1

    def getAction(self):
        currentState = torch.Tensor([self.currentState])
        QValue = self.Q_net(currentState)[0]
        action = np.zeros(self.actions)
        action_index=-1
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:         #random返回0-1的浮点数
                print("random action")
                action_index = self.getRondomAction()
                #action_index = self.randomAction[self.timeStep%2]
                self.actionIndex=action_index
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue.detach().numpy())
                self.actionIndex=action_index
                action[action_index] = 1
        else:
            action[0] = 1  # do nothing

        # change episilon
        # if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
        #     self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        print(action)
        return action

    def setInitState(self, observation):
        #self.currentState = np.stack((observation, observation, observation, observation),axis=0)
        self.currentState=[]
        for i in range(0,4):
            for j in range(0,3):
                self.currentState.append(observation[j])
        self.currentState=np.array(self.currentState)


def getScreenshot():                                                    #得到整个屏幕的截图，用cv2读取出来，返回的是一个三维数组
    screenshot = pg.screenshot(region=[0, 0, 1920, 1080])  # x,y,w,h
    screenshot.save('screenshot.png')
    cur_img = cv2.imread('screenshot.png')
    return cur_img

def MP(x, y):
    try:
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y)
    except:
        print('Move Error')

# 替换frame_step
def doAction(input_action):
    start = time.time()

    if sum(input_action) != 1:
        raise ValueError('Multiple input actions!')
    for i in range(11):
        if input_action[i] == 1:
            if i % 2 == 0:
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.3)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            tmp_i = i // 2
            if tmp_i == 1:      #
                MP(-120, 0)     #2,3  向左
            elif tmp_i == 2:
                MP(120, 0)      #4,5  向右
            elif tmp_i == 3:
                MP(0, 60)      #6,7  向下
                print("down")
            elif tmp_i == 4:
                MP(0, -70)     #8,9  向上
                print("up")
            elif tmp_i==5:              #10 转向180
                MP(240,0)
                time.sleep(0.2)
                MP(240,0)
                time.sleep(0.2)
                MP(240, 0)
                time.sleep(0.2)
                MP(240, 0)
                time.sleep(0.2)
                MP(240, 0)
                time.sleep(0.2)
                MP(240, 0)
                time.sleep(0.2)
                MP(240, 0)
                time.sleep(0.2)
                MP(240, 0)
                time.sleep(0.2)
                MP(240, 0)
                time.sleep(0.2)
                MP(240, 0)
                time.sleep(0.2)
                MP(240, 0)
                time.sleep(0.2)
                MP(240, 0)
                time.sleep(0.6)
            break
    time.sleep(0.2)
    end = time.time()


def getTerminal(hsv):                                                       #输入的hsv是整个屏幕截屏所对应的hsv
    # subImg=hsv[995:1025,1145:1155]
    # low_hsv = np.array([0, 0, 0])
    # high_hsv = np.array([180, 255, 46])
    # mask_black = cv2.inRange(subImg, lowerb=low_hsv, upperb=high_hsv)        #判断某个区域黑色占比
    # cnt=0
    # for y in range(mask_black.shape[0]):
    #     for x in range(mask_black.shape[1]):
    #         if mask_black[y][x] != 0:
    #             cnt += 1
    # #print("cnt: ",cnt/(mask_black.shape[0]*mask_black.shape[1]))
    # if(cnt/(mask_black.shape[0]*mask_black.shape[1])>0.9):
    #     return True
    #
    #
    # subImg=hsv[1025:1065,1080:1090]
    # mask_black = cv2.inRange(subImg, lowerb=low_hsv, upperb=high_hsv)       #判断另一个区域黑色占比，只有其中一个区域黑色占比过多则表明游戏结束
    # cnt = 0
    # for y in range(mask_black.shape[0]):
    #     for x in range(mask_black.shape[1]):
    #         if mask_black[y][x] != 0:
    #             cnt += 1
    # if (cnt / (mask_black.shape[0] * mask_black.shape[1]) > 0.9):
    #     return True
    subImg = hsv[985:1030, 1720:1900]
    ratio = get_ratio(subImg, np.array([10,100, 100]), np.array([31, 255, 255]), 45 * 180)
    print("terminal ratio1: ", ratio)
    if (ratio > 0.1):
        return True
    subImg = hsv[1020:1050, 1450:1630]
    ratio = get_ratio(subImg, np.array([5, 100, 100]), np.array([31, 200, 255]), 30*180)
    print("terminal ratio2: ", ratio)
    if (ratio > 0.5):
        return True
    return False


def click(x,y):
    m=PyMouse()
    m.press(x, y, 1)
    time.sleep(0.2)
    m.release(x, y, 1)
    time.sleep(0.2)

def restart():
    global GAME_STATE,red_damage,yellow_damage,orange_damage,hight,enemyInRange
    hight=1500
    enemyInRange=0
    areaRatio=0
    if(GAME_STATE==0):  #游戏未开始
        click(1750,860)  #准备
        time.sleep(1)
        click(1750,860)  #开始
        time.sleep(10)
        click(1500,1040) #加入游戏
        time.sleep(75)
        GAME_STATE=1
        red_damage=yellow_damage=orange_damage=0
        hight=1500
    elif(GAME_STATE==2): #第一条命死亡，需要重新加入游戏
        time.sleep(10)
        click(1500,1040)  #加入游戏
        time.sleep(15)
        GAME_STATE=3
    elif(GAME_STATE==4):  #第二条命也死亡，需要重开一局人机局
        time.sleep(10)
        click(1500,1040)   #返回基地
        time.sleep(5)
        click(1500,1040)   #确定
        time.sleep(6)
        click(1750, 860)   #准备
        time.sleep(1)
        click(1750, 860)   #开始
        time.sleep(10)
        click(1500, 1040)  #加入战斗
        time.sleep(75)
        GAME_STATE = 1


# 获取颜色占比
def get_ratio(hsv, low_hsv, high_hsv, total):
    cnt = 0
    mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    cv2.imwrite('test/test_mask.jpg', mask)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] != 0:
                cnt += 1
    #print(round(cnt / total, 3))
    return round(cnt / total, 3)


# 通过灰色的点判断是否在空域中
# 参数 - 雷达图片
# 返回 - 是否在区域中
def is_in_area(img_map):
    cnt = 0
    for i in range(img_map.shape[0]):
        for j in range(img_map.shape[1]):
            sum = 0
            for k in range(3):
                sum += img_map[i][j][k]
            avg = sum / 3
            if avg > 200 or avg < 50:
                for k in range(3):
                    img_map[i][j][k] = 255
                continue
            dif = 0
            for k in range(3):
                dif += abs(avg - img_map[i][j][k])
            if dif > 40:
                for k in range(3):
                    img_map[i][j][k] = 255
                continue
            cnt += 1
    # 不在区域中，灰色的点会很少，在区域中，灰色的点至少大于200个，完全在区域中大概500多个
    return cnt/1100

# 视野中是否存在敌机
def is_enemy_in_view(img, hsv):
    region = []

    # 1.  过滤截图只剩红
    low_hsv = np.array([0, 43, 46])
    high_hsv = np.array([10, 255, 255])
    mask_red = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)

    # 2. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (26, 6))

    # 3. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(mask_red, element2, iterations=1)

    # 4. 腐蚀一次，去掉细节
    erosion = cv2.erode(dilation, element1, iterations=1)

    # 5. 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=3)

    # 6. 存储中间图片 测试用
    # cv2.imwrite("pic/dilation.png", dilation)
    # cv2.imwrite("pic/erosion.png", erosion)
    # cv2.imwrite("pic/dilation2.png", dilation2)

    # 7. 查找轮廓
    contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)

        # 面积小的都筛选掉
        if area < 800:
            continue

        # 轮廓近似，作用很小
        # epsilon = 0.001 * cv2.arcLength(cnt, True)
        # approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)

        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        # print(box[0][1], box[2][1], box[0][0], box[2][0])

        # 筛选那些太细的矩形，留下扁的
        if width < 230 or width > 270 or height < 60 or height > 120:
            continue

        region.append(box)

    # 7.带轮廓的图片 测试用
    # for box in region:
    #     cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
    #
    # cv2.imwrite("contours.png", img)

    return len(region)

# 射击点旁边是否有敌机
def is_enemy_in_range(hsv):
    total = 800
    value = get_ratio(hsv, np.array([0, 43, 46]), np.array([10, 255, 255]), total)
    return value


def ocr(img_path):
    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"
    # 二进制方式打开图片文件
    f = open(img_path, 'rb')
    img = base64.b64encode(f.read())
    params = {"image": img}
    #access_token = '24.d8f80e660475de97b82a65b7084f3c92.2592000.1611560302.282335-23021910'
    request_url = request_url + "?access_token=" + accessToken
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)

    # print(response.json())
    return response.json()

# TODO
# 获取reward
def get_reward(hsv,img,timeStep,actionIndex):                                    #hsv和img都是整个屏幕的截图
    global yellow_damage,red_damage,orange_damage,hight,enemyInRange,areaRatio
    reward=0

    # 这块处理损伤
    # 飞机图标的面积大约是15000
    # total = 15000
    # hsv_damage = hsv[850:1050, 0:250]
    # new_yellow_damage = get_ratio(hsv_damage, np.array([26, 43, 46]), np.array([34, 255, 255]), total)
    # new_orange_damage = get_ratio(hsv_damage, np.array([11, 43, 46]), np.array([25, 255, 255]), total)
    # new_red_damage = get_ratio(hsv_damage, np.array([0, 43, 46]), np.array([10, 255, 255]), total)
    # reward+=(red_damage-new_red_damage)*10+(yellow_damage-new_yellow_damage)*5+(orange_damage-new_orange_damage)*2
    # red_damage=new_red_damage
    # orange_damage=new_orange_damage
    # yellow_damage=new_yellow_damage


    # 这块分析占领条
    total = (1152 - 768) * (32 - 24)
    hsv_occupy = hsv[24:32, 768:1152]
    # 蓝色占比
    # low_hsv = np.array([100, 43, 46])
    # high_hsv = np.array([124, 255, 255])
    blue_occupy_ratio = get_ratio(hsv_occupy, np.array([100, 43, 46]), np.array([124, 255, 255]),
                                  (1152 - 768) * (32 - 24)) + 0.0027
    # 红色占比
    red_occupy_ratio = get_ratio(hsv_occupy, np.array([0, 43, 46]), np.array([10, 255, 255]),
                                 (1152 - 768) * (32 - 24)) - 0.0027




    # 分析雷达，判断是否朝空域飞，判断是否在目标空域中,截取37*37
    # 灰色,不用hsv了，rgb三个值相近且均值在100左右会显示灰色
    # low_hsv = np.array([0, 0, 46])
    # high_hsv = np.array([180, 43, 220])
    img_map = img[106:125, 1771:1808]
    newAreaRatio = is_in_area(img_map)
    diff=(newAreaRatio-areaRatio)*200

    reward+=diff
    areaRatio=newAreaRatio


    if(timeStep%3==0):
        hightImg=img[90:115,100:150]
        cv2.imwrite("hight.png",hightImg)
        hightResult=ocr("hight.png").get("words_result")
        if(len(hightResult)>=1):
            hightResult=hightResult[0].get("words")
            hightResult=int(hightResult)
            hight=hightResult


    if(hight>1400):
        print("hight out of range")
        if (actionIndex == 6 or actionIndex == 7):
            reward+=15
        elif(actionIndex==8 or actionIndex==9):
            reward-=15

    elif(hight<1100):
        print("hight out of range")
        if (actionIndex == 9 or actionIndex == 8):
            reward+=15
        elif (actionIndex==6 or actionIndex==7):
            reward-=15


    # hitImg=img[235:270,920:1000]
    # cv2.imwrite("hit.png",hitImg)
    # hitResult=ocr("hit.png").get("words_result")
    # if(hitResult and len(hitResult)>=1):
    #     reward+=2

    # enemyCount=is_enemy_in_view(cur_img,hsv)
    # reward+=enemyCount
    enemyRatio=is_enemy_in_range(hsv[250:770,590:1470])
    print("enemyRatio: ",enemyRatio)
    if(enemyRatio>3 and enemyInRange==0):
        reward+=(enemyRatio-3)*60
        enemyInRange=1
    if(enemyRatio<3 and enemyInRange==1):
        reward-=30
        enemyInRange=0

    print("hight: ",hight)
    # print("orange: ",orange_damage)
    # print("yellow: ",yellow_damage)
    # print("red: ",red_damage)
    print("area ratio: ",areaRatio)
    #print("enemyCount: ", enemyCount)
    print("reward: ",reward)
    print("\n")

    return reward

if __name__ == '__main__':

    actions = 11
    brain = BrainDQNMain(actions) # Step 2: init Flappy Bird Game
    while(1!=0):
        restart()
        cur_img=getScreenshot()  #1920*1080*3
    # cur_img=[[[1,2,3],[1,2,3]],
    #          [[4,5,6],[4,5,6]]]

        cur_img = cv2.resize(cur_img, (1000, 1000))
        cur_img = np.array(cur_img)  #1000*1000*3
        cur_img = cur_img.transpose(2, 1, 0)   #3*1000*1000
        brain.setInitState(cur_img)
        print(brain.currentState.shape)  #12*1000*1000  12是通道数，RGB本身占三个通道，去连续四针图片作为一个state，共4*3=12个通道

    # brain.storeToMemory([[[6,6],[5,5]],[[4,4],[3,3]],[[2,2],[1,1]]])



        while 1!= 0:
            action = brain.getAction()  #类似于[1.,0.]的一维数组
            #print("action ",action)
        #cur_img = getScreenshot()  # 1920*1080*3
            doAction(action)
            cur_img=getScreenshot()
            hsv = cv2.cvtColor(cur_img, cv2.COLOR_BGR2HSV)
            terminal=getTerminal(hsv)
            if(terminal):
                GAME_STATE+=1
                print("terminal!!\n\n")
                print("state ",GAME_STATE)
                break
            reward = get_reward(hsv,cur_img,brain.timeStep,brain.actionIndex)
            cur_img = np.array(cur_img)
            cur_img = cv2.resize(cur_img, (1000, 1000))
            cur_img = np.array(cur_img)  # 1000*1000*3
            cur_img = cur_img.transpose(2, 1, 0)  # 3*1000*1000
            brain.storeToMemory(action,cur_img,reward,terminal)
            time.sleep(0.1)
            print("timeStep: ",brain.timeStep)
            # if(brain.saveCount==1):
            #     sys.exit(0)