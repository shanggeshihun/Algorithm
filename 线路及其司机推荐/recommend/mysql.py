# -*- coding: utf-8 -*-
"""
Created on 20190510

"""
# coding:utf-8
import pymysql
import configparser
class mysql_operation():
    def __init__(self):
        cfg_file=r"E:\SJB\NOTE\Python\algorithm\线路及其司机推荐\recommend\conf.ini"
        config=configparser.ConfigParser()
        config.read(cfg_file)
        self.data=config.sections()
        self.host=config.get('mysql_204','host')
        self.port=config.getint('mysql_204','port')
        self.user=config.get('mysql_204','user')
        self.passwd=config.get('mysql_204','passwd')
        self.db=config.get('mysql_204','db')
        self.charset=config.get('mysql_204','charset')

        # self.charset=config.get('mysql','charset')
        self.conn=pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            passwd=self.passwd,
            db=self.db,
            charset=self.charset
            )
        self.cursor=self.conn.cursor()

    def create(self,sql):
        try:
            self.cursor.execute(sql)
        except pymysql.Error as e:
            print(e)

    def select(self,sql):
        try:
            self.cursor.execute(sql)
        except pymysql.Error as e:
            print(e)
        else:
            self.cursor.execute(sql)
            result=self.cursor.fetchall()
        return result

    def insert(self,table,tuple_value):
        sql='insert into %s values %s' % (table,tuple_value)
        self.cursor.execute(sql)
        self.conn.commit()
#        return self.cursor.lastrowid
        
    def delete(self,sql):
        self.cursor.execute(sql)
        self.conn.commit()
        return self.affected_num()

    def affected_num(self):
        return self.cursor.rowcount

    def close(self):
        if self.cursor!=None:
            self.cursor.close()
        if self.conn!=None:
            self.conn.close()