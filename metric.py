import cv2
import numpy as np
import utils
'''
ans = (imgs[0] >= 0.5).astype(np.uint8) * 255
pred = (imgs[1] >= 0.5).astype(np.uint8) * 255
for img in imgs:
    img = (img >= 0.5).astype(np.uint8) * 255
    print(np.mean(img))
    connectivity = 4
    output = cv2.connectedComponentsWithStats(img, connectivity)
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]
    print('--------')
    print(num_labels)
    cv2.imshow('i',img); cv2.waitKey(0)
    for i in range(num_labels):
        print('label is', i)
        cv2.imshow('i',(labels==i).astype(np.uint8)*255); cv2.waitKey(0)
    print(np.unique(labels))
    print(labels.dtype)
'''

def label_objects(img):
    #print(img.shape)
    outputs = cv2.connectedComponentsWithStats(img, 4)
    labeled = outputs[1]
    return labeled

def intersection_table(ans, num_ans_labels, pred, num_pred_labels):
    '''
    row: answer label
    col: prediction label
    content: number of pixels in intersection between ans and pred  

    example
    *   0   1   2   3  <- prediction label 
    0   0   0   0   0
    1   0   2   87  1
    2   0   23  4   9
    3   0   0   0   34
    ^-- answer label

    NOTE 
    Table always has label 0. But it's meaningless.
    (label 0 is background, )
    Just for convenient indexing.
    '''
    assert ans.shape == pred.shape
    itab = np.zeros((num_ans_labels,num_pred_labels),dtype=int)
    #print(itab.shape, num_ans_labels, num_pred_labels)
    for ans_label in range(1,num_ans_labels):
        for pred_label in range(1,num_pred_labels):
            ans_component = (ans == ans_label)
            pred_component = (pred == pred_label)
            intersection = ans_component * pred_component
            num_intersected = np.sum(intersection.astype(int))
            #print(ans_label,pred_label)
            itab[ans_label,pred_label] = num_intersected
    #print(itab)
    return itab

def tp_table(itab):
    '''
    itab: intersection_table
    return true-positive value table
    '''
    def leave_max(v):
        m = np.max(v)
        i = np.argmax(v)
        v[:] = 0
        v[i] = m
        return v
    ret_tab = itab.copy()
    ret_tab = np.array(list(map(leave_max, ret_tab)))
    ret_tab = np.array(list(map(leave_max, ret_tab.T)))
    ret_tab = ret_tab.T
    return ret_tab

def confusion_stats(tp_table):
    len_y = len(tp_table)
    len_x = len(tp_table[0])
    ys = np.argmax(tp_table, axis=0)
    xs = np.argmax(tp_table, axis=1)

    tp = len(np.unique(ys)) - 1 # skip 0
    fp = len_x - tp - 1 # skip 0
    fn = len_y - tp - 1 # skip 0
    '''
    print(tp,fp,fn)
    print('yi:',ys)
    print('xi:',xs)
    ys = filter(lambda y: y != 0,ys[1:])
    xs = filter(lambda x: x != 0,xs[1:])
    tp_yxs = [(0,0)] + list(zip(ys,xs))
    '''
    tp_yxs = [(0,0)]
    for y in range(1,len_y):
        for x in range(1,len_x):
            if tp_table[y][x] != 0:
                tp_yxs.append( (y,x) )
                # Stack (y,x) into tp_yxs.
                # Verify early test cases.
                # Add more test cases..

    return tp,fp,fn, tp_yxs

def intersection_areas(tp_table, tp_yxs):
    return list(map(lambda yx: tp_table[yx], tp_yxs))

def f1score(tp, fp, fn):
    precision = tp / (tp + fp) if tp != 0 else 0
    recall = tp / (tp + fn) if tp != 0 else 0
    if tp != 0:
        return 2 * precision * recall / (precision + recall)
    else:
        return 0

def area_ratios(areas, sum_areas):
    def ratio(area):
        return area / sum_areas if sum_areas != 0 else 0.
    return list(map(ratio, areas))

def object_dice(tp_tab, tp_yxs, ans_areas, pred_areas):
    '''
    tp_tab: true positive area table, y=ans_label, x=pred_label
    tp_yxs: true positive label pair list<ans_label,pred_label>
    '''
    intersections = intersection_areas(tp_tab, tp_yxs)
    #print('intersections:',intersections)
    ans_areas[0] = 0
    pred_areas[0] = 0
    gamma = area_ratios(ans_areas,sum(ans_areas))
    sigma = area_ratios(pred_areas,sum(pred_areas))
    #print('g',gamma)
    #print('s',sigma)

    g_dice = 0 # ground truth dice 
    s_dice = 0 # segmented dice
    for i,(y,x) in enumerate(tp_yxs[1:],1):
        dice = 2 * intersections[i] / (ans_areas[y] + pred_areas[x])
        g_dice += gamma[y] * dice
        s_dice += sigma[x] * dice
    #print(g_dice, s_dice)
    return float(0.5 * (g_dice + s_dice))

def my_old_metric(ans,pred):
    ans = (ans >= 0.5).astype(np.uint8) * 255
    pred = (pred >= 0.5).astype(np.uint8) * 255
    ans_ouput = cv2.connectedComponentsWithStats(ans, 4)
    ans = ans_ouput[1]
    ans_areas = ans_ouput[2][:,cv2.CC_STAT_AREA]
    #print(type(ans),np.unique(ans))
    #cv2.imshow('ans',ans); cv2.waitKey(0)
    #print(' ans:',ans_areas)
    #for i in range(len(ans_areas)): cv2.imshow('ans', (ans == i).astype(np.uint8) * 255); cv2.waitKey(0)

    pred_ouput = cv2.connectedComponentsWithStats(pred, 4)
    pred = pred_ouput[1]
    pred_areas = pred_ouput[2][:,cv2.CC_STAT_AREA]
    #print(type(pred),np.unique(pred))
    #print('pred:',pred_areas)
    #for i in range(len(pred_areas)): cv2.imshow('pred', (pred == i).astype(np.uint8) * 255); cv2.waitKey(0)

    tp_tab = tp_table(intersection_table(ans,len(ans_areas), 
                                         pred,len(pred_areas)))
    #print('tp_tab\n',tp_tab)
    tp, fp, fn, tp_yxs = confusion_stats(tp_tab)
    #print('tp_yxs', tp_yxs)
    f1 = f1score(tp,fp,fn)
    dice_obj = object_dice(tp_tab, tp_yxs, ans_areas,pred_areas)

    return f1, dice_obj

from oct2py import Oct2Py
#oct2py = Oct2Py(temp_dir='/run/shm')#ubuntu
oct2py  = Oct2Py(temp_dir='/tmp')#ubuntu: use tmpfs...
def advanced_metric(ans, pred):
    #cv2.imshow('pred', pred)
    #cv2.imshow('ans', ans); cv2.waitKey(0)
    ans = (ans >= 0.5).astype(np.uint8) * 255
    pred = (pred >= 0.5).astype(np.uint8) * 255

    S = label_objects(pred)
    G = label_objects(ans)

    #f1_v1 = oc.F1score_v1(S,G)
    f1_v2 = oct2py.F1score_v2(S,G)
    dice_obj = oct2py.ObjectDice(S,G)
    #hausdorff_obj = oc.ObjectHausdorff(S,G)

    #return f1_v1, dice_obj#, hausdorff_obj
    return f1_v2, dice_obj#, hausdorff_obj
    '''
    #cv2.imshow('ans',ans)
    #cv2.imshow('pred',pred); cv2.waitKey(0)

    ans_ouput = cv2.connectedComponentsWithStats(ans, 4)
    ans = ans_ouput[1]
    ans_areas = ans_ouput[2][:,cv2.CC_STAT_AREA]
    #print(type(ans),np.unique(ans))
    #cv2.imshow('ans',ans); cv2.waitKey(0)
    #print(' ans:',ans_areas)
    for i in range(len(ans_areas)): cv2.imshow('ans', (ans == i).astype(np.uint8) * 255); cv2.waitKey(0)

    pred_ouput = cv2.connectedComponentsWithStats(pred, 4)
    pred = pred_ouput[1]
    pred_areas = pred_ouput[2][:,cv2.CC_STAT_AREA]
    #print(type(pred),np.unique(pred))
    #print('pred:',pred_areas)
    #for i in range(len(pred_areas)): cv2.imshow('pred', (pred == i).astype(np.uint8) * 255); cv2.waitKey(0)

    tp_tab = tp_table(intersection_table(ans,len(ans_areas), 
                                         pred,len(pred_areas)))
    #print('tp_tab\n',tp_tab)
    tp, fp, fn, tp_yxs = confusion_stats(tp_tab)
    #print('tp_yxs', tp_yxs)
    f1 = f1score(tp,fp,fn)
    dice_obj = object_dice(tp_tab, tp_yxs, ans_areas,pred_areas)

    return f1, dice_obj
    '''
'''
cv2.imshow('a',ans_component.astype(np.float32))
cv2.imshow('p',pred_component.astype(np.float32))
cv2.imshow('i',intersection.astype(np.float32)); cv2.waitKey(0)
print(ans_component.astype(int))
print(pred_component.astype(int))
print(intersection.astype(int))
'''
    
import unittest
class TestMetrics(unittest.TestCase):
    def test_real_data1(self):
        print('------ two similar images -----')
        pred = cv2.imread('./img/train_21_predict.bmp',0)
        ans = cv2.imread('./img/train_21_anno.bmp',0)

        S = label_objects(pred)
        G = label_objects(ans)

        oc = Oct2Py()
        f1_v1 = oc.F1score_v1(S,G)
        f1_v2 = oc.F1score_v2(S,G)
        dice_obj = oc.ObjectDice(S,G)
        hausdorff_obj = oc.ObjectHausdorff(S,G)

        #f1, dice_obj = advanced_metric(ans,pred)
        print('f1score v1 =', f1_v1)
        print('f1score v2 =', f1_v2)
        print('dice_obj =', dice_obj)
        print('hausdorff_obj =', hausdorff_obj)

    def test_real_data2(self):
        print('------ two similar images -----')
        pred = cv2.imread('./img/testA_13_predict.bmp',0)
        ans = cv2.imread('./img/testA_13_anno.bmp',0)
        #print(type(pred))
        #print(pred)
        #cv2.imshow('pred',pred); cv2.waitKey(0)

        S = label_objects(pred)
        G = label_objects(ans)

        oc = Oct2Py()
        f1_v1 = oc.F1score_v1(S,G)
        f1_v2 = oc.F1score_v2(S,G)
        dice_obj = oc.ObjectDice(S,G)
        hausdorff_obj = oc.ObjectHausdorff(S,G)
        #hausdorff_obj = oc.ObjectHausdorff(S,G)

        #f1, dice_obj = advanced_metric(ans,pred)
        print('f1score v1 =', f1_v1)
        print('f1score v2 =', f1_v2)
        print('dice_obj =', dice_obj)
        print('hausdorff_obj =', hausdorff_obj)
'''
class test_itable(unittest.TestCase):
    def test_unmatched_shape(self):
        ans = np.ones((3,2))
        pred = np.ones((2,2))
        self.assertRaises(AssertionError, 
                          intersection_table, ans, 0, pred, 0)

    def test_simple_case(self):
        ans = np.array([
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,1,1,1,0,0,0,0,0,0,0],
            [0,1,1,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,2,2,0,0,0,0,0,0,0],
            [0,0,2,2,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
        ],dtype=np.uint8)
        pred = np.array([
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,1,0,2,2,0,3,3,0],
            [0,0,1,1,0,2,2,0,3,3,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
        ],dtype=np.uint8)
        expected_itab = np.array([
            [0,0,0,0],
            [0,4,0,0],
            [0,0,0,0],
        ],dtype=int)

        #cv2.imshow('a',ans.astype(np.float32))
        #cv2.imshow('p',pred.astype(np.float32)); cv2.waitKey(0)
        actual_itab = intersection_table(ans,len(np.unique(ans)), 
                                         pred,len(np.unique(pred)))
        self.assertEqual(actual_itab.tolist(),
                         expected_itab.tolist())

    def test_many_value_in_1row_case(self):
        ans = np.array([
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,1,1,1,1,0],
            [0,2,2,2,0,1,1,1,1,1,0],
            [0,2,2,2,0,1,1,1,1,1,0],
            [0,2,2,2,0,1,1,1,1,1,0],
            [0,0,0,0,0,1,1,1,1,1,0],
            [0,0,0,0,0,1,1,1,1,1,0],
            [0,3,3,0,0,0,0,0,0,0,0],
            [0,3,3,0,4,4,0,0,0,0,0],
            [0,0,0,0,0,4,0,0,0,5,5],
            [0,0,0,0,0,0,0,0,0,5,5],
        ],dtype=np.uint8)
        pred = np.array([
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,1,0,0,0,0],
            [0,0,0,0,0,1,1,0,2,2,0],
            [0,3,3,3,0,0,0,0,2,0,0],
            [0,3,3,3,0,0,0,0,0,0,0],
            [0,3,3,3,0,0,4,4,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,5,5,5,5,5,0,0,0,0,0],
            [0,5,5,5,5,5,0,0,0,0,0],
            [0,5,5,5,5,5,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
        ],dtype=np.uint8)
        expected_itab = np.array([
            [0,0,0,0,0,0],
            [0,4,3,0,2,0],
            [0,0,0,9,0,0],
            [0,0,0,0,0,4],
            [0,0,0,0,0,3],
            [0,0,0,0,0,0],
        ],dtype=int)

        actual_itab = intersection_table(ans,len(np.unique(ans)), 
                                         pred,len(np.unique(pred)))

class Test_stats(unittest.TestCase):
    def test_max_itab(self):
        itab = np.array([
            [0,0,0,0,0,0],
            [0,4,3,0,2,0],
            [0,0,0,9,0,0],
            [0,0,0,0,0,4],
            [0,0,0,0,0,3],
        ],dtype=int)
        expected = np.array([
            [0,0,0,0,0,0],
            [0,4,0,0,0,0],
            [0,0,0,9,0,0],
            [0,0,0,0,0,4],
            [0,0,0,0,0,0],
        ],dtype=int)
        actual = tp_table(itab)

        self.assertEqual(actual.tolist(), expected.tolist())

    def test_stats(self):
        tp_tab = np.array([
            [0,0,0,0,0,0],
            [0,4,0,0,0,0],
            [0,0,0,9,0,0],
            [0,0,0,0,0,4],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
        ],dtype=int)
        expected = (3, 2, 2)
        tp, fp, fn, tp_yxs = confusion_stats(tp_tab)
        self.assertEqual((tp,fp,fn), expected)
        self.assertEqual(tp_yxs, [(0,0),(1,1),(2,3),(3,5)])

        #print('f1score =',f1score(*expected))

    def test_dice_obj(self):
        ans = np.array([
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,1,1,1,1,0],
            [0,2,2,2,0,1,1,1,1,1,0],
            [0,2,2,2,0,1,1,1,1,1,0],
            [0,2,2,2,0,1,1,1,1,1,0],
            [0,0,0,0,0,1,1,1,1,1,0],
            [0,0,0,0,0,1,1,1,1,1,0],
            [0,3,3,0,0,0,0,0,0,0,0],
            [0,3,3,0,4,4,0,0,0,0,0],
            [0,0,0,0,0,4,0,0,0,5,5],
            [0,0,0,0,0,0,0,0,0,5,5],
        ],dtype=np.uint8)
        pred = np.array([
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,1,0,0,0,0],
            [0,0,0,0,0,1,1,0,2,2,0],
            [0,3,3,3,0,0,0,0,2,0,0],
            [0,3,3,3,0,0,0,0,0,0,0],
            [0,3,3,3,0,0,4,4,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,5,5,5,5,5,0,0,0,0,0],
            [0,5,5,5,5,5,0,0,0,7,7],
            [0,5,5,5,5,5,0,6,0,0,0],
            [0,0,0,0,0,0,0,6,0,0,0],
        ],dtype=np.uint8)
        ans_areas = [ 0,30, 9, 4, 3, 4]
        pred_areas= [ 0, 4, 3, 9, 2,15, 2, 2]

        tp_tab = tp_table(intersection_table(ans,len(ans_areas), 
                                             pred,len(pred_areas)))
        tp, fp, fn, tp_yxs = confusion_stats(tp_tab)

        expected_stats = 3,4,2, [(0,0),(1,1),(2,3),(3,5)]
        self.assertEqual((tp, fp, fn, tp_yxs), expected_stats)
        print(f1score(tp,fp,fn))

        dice_obj = object_dice(tp_tab,tp_yxs,ans_areas,pred_areas)
        self.assertAlmostEqual(dice_obj, 0.3265785289933897)
        print(dice_obj)
        # segmented dice 

    def test_0dice(self):
        ans = np.array([
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,1,1,1,1,0],
            [0,2,2,2,0,1,1,1,1,1,0],
            [0,2,2,2,0,1,1,1,1,1,0],
            [0,2,2,2,0,1,1,1,1,1,0],
            [0,0,0,0,0,1,1,1,1,1,0],
            [0,0,0,0,0,1,1,1,1,1,0],
            [0,3,3,0,0,0,0,0,0,0,0],
            [0,3,3,0,4,4,0,0,0,0,0],
            [0,0,0,0,0,4,0,0,0,5,5],
            [0,0,0,0,0,0,0,0,0,5,5],
        ],dtype=np.uint8)
        pred = np.array([
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,0,0],
            [0,2,2,0,0,0,0,1,1,0,0],
        ],dtype=np.uint8)
        ans_areas = [ 0,30, 9, 4, 3, 4]
        pred_areas = [0, 4, 2]

        tp_tab = tp_table(intersection_table(ans,len(ans_areas), 
                                             pred,len(pred_areas)))
        tp, fp, fn, tp_yxs = confusion_stats(tp_tab)

        dice_obj = object_dice(tp_tab,tp_yxs,ans_areas,pred_areas)
        self.assertAlmostEqual(dice_obj, 0)

    def test_1dice(self):
        ans = np.array([
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,1,1,1,1,0],
            [0,2,2,2,0,1,1,1,1,1,0],
            [0,2,2,2,0,1,1,1,1,1,0],
            [0,2,2,2,0,1,1,1,1,1,0],
            [0,0,0,0,0,1,1,1,1,1,0],
            [0,0,0,0,0,1,1,1,1,1,0],
            [0,3,3,0,0,0,0,0,0,0,0],
            [0,3,3,0,4,4,0,0,0,0,0],
            [0,0,0,0,0,4,0,0,0,5,5],
            [0,0,0,0,0,0,0,0,0,5,5],
        ],dtype=np.uint8)
        pred = ans.copy()
        ans_areas = pred_areas = [ 0,30, 9, 4, 3, 4]
        tp_tab = tp_table(intersection_table(ans,len(ans_areas), 
                                             pred,len(pred_areas)))
        tp, fp, fn, tp_yxs = confusion_stats(tp_tab)

        dice_obj = object_dice(tp_tab,tp_yxs,ans_areas,pred_areas)
        self.assertAlmostEqual(dice_obj, 1)

    def test_1label(self):
        ans = np.array([
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,1,1,1,1,0],
            [0,0,0,0,0,1,1,1,1,1,0],
            [0,0,0,0,0,1,1,1,1,1,0],
            [0,0,0,0,0,1,1,1,1,1,0],
            [0,0,0,0,0,1,1,1,1,1,0],
            [0,0,0,0,0,1,1,1,1,1,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0],
        ],dtype=np.uint8)
        pred = ans.copy()
        ans_areas = pred_areas = [0,30]
        tp_tab = tp_table(intersection_table(ans,len(ans_areas), 
                                             pred,len(pred_areas)))
        tp, fp, fn, tp_yxs = confusion_stats(tp_tab)

        dice_obj = object_dice(tp_tab,tp_yxs,ans_areas,pred_areas)
        self.assertAlmostEqual(dice_obj, 1)

    #@unittest.skip('later')
    def test_real_data_same_img(self):
        print('--------------- same imgs ---------------')
        ans = cv2.imread('./img/0ans.png',0)
        pred = ans.copy()

        f1, dice_obj = advanced_metric(ans,pred)

        self.assertEqual(f1,1.0)
        self.assertEqual(dice_obj,1.0)

    def test_real_data_pred_no_label_case(self):
        print('------ ordinary ans, no label pred -----')
        ans = cv2.imread('./img/0ans.png',0)
        pred = np.zeros(ans.shape)

        f1, dice_obj = advanced_metric(ans,pred)

        self.assertEqual(f1,0.0)
        self.assertEqual(dice_obj,0.0)

    def test_real_data_ans_no_label_case(self):
        print('------ no label ans, ordinary pred -----')
        pred = cv2.imread('./img/0ans.png',0)
        ans = np.zeros(pred.shape)

        f1, dice_obj = advanced_metric(ans,pred)

        self.assertEqual(f1,0.0)
        self.assertEqual(dice_obj,0.0)

    @unittest.skip('manual test')
    def test_real_data0(self):
        print('------ ordinary ans, pred has big FN -----')
        ans = cv2.imread('./img/0ans.png',0)
        pred = cv2.imread('./img/0pred.png',0)

        f1, dice_obj = advanced_metric(ans,pred)
        print('f1score =', f1)
        print('dice_obj =', dice_obj)

    @unittest.skip('manual test')
    def test_real_data1(self):
        print('------ ordinary ans, pred has small FN -----')
        ans = cv2.imread('./img/1ans.png',0)
        pred = cv2.imread('./img/1pred.png',0)

        f1, dice_obj = advanced_metric(ans,pred)
        print(type(f1),type(dice_obj))
        print('f1score =', f1)
        print('dice_obj =', dice_obj)

    @unittest.skip('manual test')
    def test_real_data2(self):
        print('------ ordinary ans, pred has small FN -----')
        ans = cv2.imread('./img/2ans.png',0)
        pred = cv2.imread('./img/2pred.png',0)

        f1, dice_obj = advanced_metric(ans,pred)
        print('f1score =', f1)
        print('dice_obj =', dice_obj)

    #@unittest.skip('manual test')
    def test_real_bio_data(self):
        print('------ two similar images -----')
        ans = cv2.imread('./img/train_21_anno.bmp',0)
        pred = cv2.imread('./img/train_21_predict.bmp',0)

        f1, dice_obj = advanced_metric(ans,pred)
        print('f1score =', f1)
        print('dice_obj =', dice_obj)
'''

if __name__ == '__main__':
    unittest.main()
