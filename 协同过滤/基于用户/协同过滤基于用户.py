# -*- coding: UTF-8 -*-
from math import sqrt,pow
import operator

#基于用户的协同过滤推荐类
class UserCf():
    #获得初始化数据
    def __init__(self,data):                                      #构造函数，data为数据
        self.data = data                                            #类的属性等于数据

    #计算用户1、用户2两个用户的皮尔逊相关系数
    def pearson(self,user1,user2):#数据格式为：电影，评分
        sumXY = 0.0                                                 #初始化计算XY
        n     = 0                                                   #数量
        sumX  = 0.0                                                 #初始化计算X的总和
        sumY  = 0.0                                                 #初始化计算Y的总和
        sumX2 = 0.0                                                 #初始化计算X平方的总和
        sumY2 = 0.0                                                 #初始化计算Y平方的总和
        try:
            for movie1,score1 in user1.items():                   #轮询其中一个用户的电源列表，获得电影名和评分
                if movie1 in user2.keys():                        #如果该电影也在另一个用户的名单里
                    n += 1                                        #数量加1
                    sumXY += score1 * user2[movie1]               #计算XY，就是两个人的分数相乘并且累加
                    sumX  += score1                               #计算X的累加和
                    sumY  += user2[movie1]                        #计算Y的累加和
                    sumX2 += pow(score1,2)                        #计算X平方的累加和
                    sumY2 += pow(user2[movie1],2)                 #计算Y平方的累加和

            #计算sum(XY) - ((sum(X)*sum(Y))/n)
            molecule = sumXY-(sumX * sumY)/n
            #计算分母
            denominator = sqrt((sumX2 - pow(sumX,2) / n) * (sumY2 - pow(sumY,2) / n))
            #计算最后的pearson相关系数
            r = molecule / denominator
        except Exception as e:
            print("异常信息:",e)
            return None
        return r

    #计算与当前用户的距离，获得最临近的用户，n表示前n个
    def nearstUser(self,username,n = 1):
        distances={}                                               #用于存储用户和相关的相似度
        for otherUser,items in self.data.items():                  #遍历整个数据集
            if otherUser not in username:                          #非当前的用户
                #计算pearson相关系数
                distance = self.pearson(self.data[username],self.data[otherUser])#计算两个用户的相似度
                distances[otherUser]=distance                      #存储该用户与目标用户的相关系数
        #对用户按照pearson相关系数进行降序排序
        sortedDistance = sorted(distances.items(),key=operator.itemgetter(1),reverse=True)#最相似的N个用户
        print("排序后的用户为：",sortedDistance)
        #返回前n个用户，顾头不顾尾
        return sortedDistance[:n]


    #将当前用户没有看过的电影，按照评分从高到低推荐给客户
    #给用户推荐电影  n为推荐的个数
    def recomand(self,username,n = 1):
        recommand = {}                                                #待推荐的电影
        #轮询最相近的n个用户，取得用户名和pearson相关系数
        for user,score in dict(self.nearstUser(username,n)).items():#最相近的n个用户
            print("推荐的用户：",(user,score))
            for movies,scores in self.data[user].items():           #获得推荐的用户的电影列表
                if movies not in self.data[username].keys():        #如果当前电影当前用户没有看过没有看过
                    print("%s为该用户推荐的电影：%s"%(user,movies))
                    if movies not in recommand.keys():              #并且不存在推荐列表中
                        recommand[movies] = scores                  #将电影放入推荐列表

        #对当前推荐列表按照评分的高低进行降序排序
        sortedDistance = sorted(recommand.items(),key=operator.itemgetter(1),reverse=True)#对推荐的结果按照电影评分排序
        #返回前n个给到用户
        return sortedDistance[:n]

if __name__=='__main__':
    users = {'张三': {'速度与激情8': 2.5, '千与千寻': 3.5,
                           '阿拉丁': 3.0, '无所不能': 3.5, '无问东西': 2.5,
                           '流浪地球': 3.0},

             '李四': {'速度与激情8': 3.0, '千与千寻': 3.5,
                              '阿拉丁': 1.5, '无所不能': 5.0, '流浪地球': 3.0,
                              '无问东西': 3.5},

             '王五': {'速度与激情8': 2.5, '千与千寻': 3.0,
                                  '无所不能': 3.5, '流浪地球': 4.0},

             '马六': {'千与千寻': 3.5, '阿拉丁': 3.0,
                              '流浪地球': 4.5, '无所不能': 4.0,
                              '无问东西': 2.5},

             '刘八': {'速度与激情8': 3.0, '千与千寻': 4.0,
                              '阿拉丁': 2.0, '无所不能': 3.0, '流浪地球': 3.0,
                              '无问东西': 2.0},

             '杨九': {'速度与激情8': 3.0, '千与千寻': 4.0,
                               '流浪地球': 3.0, '无所不能': 5.0, '无问东西': 3.5},

             '万万': {'千与千寻': 4.5, '无问东西': 1.0, '无所不能': 4.0}
             }

    #初始化对象，并且进行对象属性数据建立
    userCf = UserCf(data = users)
    recommandList = userCf.recomand('万万', 2)
    print("最终推荐：%s"%recommandList)
