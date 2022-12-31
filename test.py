 
def sortDets(pos_results):
    # 确定行数,list中一个元素(也是一个list类型)保存某行的所有检测结果
    list = []
    thresh = (pos_results[0][1][3]-pos_results[0][1][1])/2
    newLine = []
    newLine.append(pos_results[0])
    list.append(newLine)
    for i in range(1, len(pos_results)):
        flag = False
        for j in range(0,len(list)):
            if -thresh < pos_results[i][1][1] - list[j][0][1][1] and pos_results[i][1][1] - list[j][0][1][1] < thresh:
                list[j].append(pos_results[i])
                flag = True
                break
        if flag == False:
            newLine = []
            newLine.append(pos_results[i])
            list.append(newLine)
    # print len(list)
    # for ench in list:
    #     print ench
 
    # 每行文本位置按照x1的坐标进行排序
    for enchLine in list:
        for i in range(1,len(enchLine)):
            exchange = False
            for j in range(0,len(enchLine)-i):
                if enchLine[j][1][0] > enchLine[j+1][1][0]:
                    tmp = enchLine[j+1]
                    enchLine[j + 1] = enchLine[j]
                    enchLine[j] = tmp
                    exchange = True
            if exchange == False: #某趟排序没有交换，则已排好序
                break
    # print "------------------------------------"
    # for ench in list:
    #     print ench
 
    # 根据y1的坐标进行行排序
    for i in range(1, len(list)):
        exchange = False
        for j in range(0, len(list) - i):
            if list[j][0][1][1] > list[j + 1][0][1][1]:
                tmp = list[j + 1]
                list[j + 1] = list[j]
                list[j] = tmp
                exchange = True
        if exchange == False:
            break
 
    # print "-----------------------------------"
    # for ench in list:
    #     print ench
    return list
 
 
if __name__ == '__main__':
    pos_results = [['text', [343, 505, 453, 538], 0.9998828], ['text', [343, 378, 452, 412], 0.99986994],
                   ['text', [342, 249, 451, 287], 0.9998646], ['text', [303, 442, 409, 478], 0.9998516],
                   ['text', [345, 311, 450, 349], 0.9998467], ['text', [111, 505, 200, 542], 0.9998406],
                   ['text', [112, 188, 193, 225], 0.99982774], ['text', [213, 186, 286, 225], 0.99982613],
                   ['text', [70, 379, 139, 415], 0.99981505], ['text', [25, 442, 97, 481], 0.9997975],
                   ['text', [295, 186, 387, 223], 0.9997931], ['text', [110, 442, 196, 478], 0.99979013],
                   ['text', [25, 509, 99, 544], 0.9997675], ['text', [151, 125, 237, 161], 0.9997596],
                   ['text', [221, 505, 3027, 542], 0.99973446], ['text', [23, 125, 94, 162], 0.9997311],
                   ['text', [150, 315, 235, 350], 0.9997166], ['text', [217, 439, 287, 478], 0.9997166],
                   ['text', [70, 315, 136, 350], 0.99971324], ['text', [23, 62, 93, 98], 0.99970144],
                   ['text', [149, 62, 238, 99], 0.9996984], ['text', [152, 378, 236, 413], 0.99969053],
                   ['text', [257, 312, 326, 351], 0.99968374], ['text', [257, 376, 327, 414], 0.9996836],
                   ['text', [26, 189, 95, 227], 0.9996604], ['text', [23, 3, 94, 38], 0.99963],
                   ['text', [132, 251, 242, 289], 0.99956113], ['text', [25, 253, 109, 288], 0.99950695],
                   ['text', [152, 2, 240, 36], 0.99950576], ['text', [255, 249, 327, 287], 0.9988224],
                   ['text', [109, 60, 134, 99], 0.9964604], ['text', [25, 379, 53, 414], 0.9940532],
                   ['text', [110, 122, 140, 161], 0.9365695]]
    list = sortDets(pos_results)  # 返回的list为已经将检测结果pos_results从左至右从上至下排好序的二维列表
    for ench in list:
        print( ench)