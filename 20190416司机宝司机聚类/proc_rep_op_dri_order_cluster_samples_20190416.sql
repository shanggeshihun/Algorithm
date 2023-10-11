-- rep_op_dri_rfm_login
DROP PROCEDURE IF EXISTS repm.proc_rep_op_dri_order_and_login_cluster_samples;
CREATE PROCEDURE repm.proc_rep_op_dri_order_and_login_cluster_samples(IN ETL_DATE VARCHAR(8))
BEGIN
 /*===============================================================+
         版权信息：版权所有(c) 2017，物易云通
         作业名称：司机聚类之运单相关样本数据
         责任人  : 
         版本号  : v1.0.0.0
         目标表  : repm.rep_op_dri_order_and_login_cluster_samples
         备注    :

         修改历史:
         版本     更改日期                      更改人             更改说明
    -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
         v1.0.0.0 2019-03-04 09:49                 liuan                   生成代码
+===============================================================*/

DECLARE `CODE` CHAR(5) DEFAULT '00000';
DECLARE msg TEXT;
DECLARE START_TIME DATETIME;
DECLARE END_TIME DATETIME;
DECLARE CONTINUE HANDLER FOR SQLEXCEPTION
    BEGIN
        /*通过GET DIAGNOSTICS这样的方式获取sqlstate message_text(当然错误信息错误编号可以自己定义)*/
        GET DIAGNOSTICS CONDITION 1 CODE = RETURNED_SQLSTATE,
        msg = MESSAGE_TEXT;

    END;
SET @ETL_DATE = STR_TO_DATE(ETL_DATE,'%Y%m%d');
/*定义时间*/
SET START_TIME = NOW();


DROP TABLE IF EXISTS test.tmp_assure_order_nocancel;
CREATE TABLE  test.tmp_assure_order_nocancel
(
`system_source_cd` VARCHAR(10) NOT NULL COMMENT '源系统代码',
`data_dt` DATE NOT NULL COMMENT '批量日期',
`sjb_ptflag` VARCHAR(2) NOT NULL COMMENT '分区标志',
`order_number` VARCHAR(32) NOT NULL COMMENT '订单号',
`driver` VARCHAR(32) DEFAULT NULL COMMENT '司机',
`driver_name` VARCHAR(32) DEFAULT NULL COMMENT '司机姓名',
`company_code` VARCHAR(32) DEFAULT NULL COMMENT '公司编码',
`pay_by_intermediary` BIGINT(20) DEFAULT '0' COMMENT '中介已支付费',
`sign_upload_time` DATETIME DEFAULT NULL COMMENT '货地签到时间',
`sign_in_state` INT(2) DEFAULT NULL COMMENT '签到状态(0:等待司机在卸货地签到; 1:装货地签到; 2:卸货地签到; 3:客户收货签到; 4:首付款协商扣款; 5:调度确认订单完成; 6:司机提交回单; 7:尾付款协商扣款; 8:调度确认收到回单; 9:已装货,待付油卡钱)',
`dynamic_state` INT(3) DEFAULT '0' COMMENT '订单动态状态(1 司机刚刚抢到单;  2 没付款之前，司机退出交易;  3 司机付款，等待中介答复;  4 司机取消订单，中介还没达成协议，司机已付款;  5 中介退出，没付款之前;  6 中介付款;  @deprecated 7 合作中，司机取消交易，协商赔付;  @deprecated 8 合作中，中介取消交易，协商赔付;  @deprecated 9 退款中;  10 订单完结;  @deprecated 11 调度点击确认完成，等待司机确认 ;  12 订单失效;  @deprecated 13 司机投诉中 ;  @deprecated 14 调度投诉;  15 司机评论;  16调度评论;  17 都评论过;  @deprecated 20 司机付款中;  @deprecated 21 调度付款中;  22 等待调度付款)',
`create_date` DATETIME DEFAULT NULL COMMENT '订单创建时间',
`agent_code` VARCHAR(32) DEFAULT NULL COMMENT '经纪人编码',
trans_days INT DEFAULT NULL COMMENT '运输天数'
);
 
INSERT INTO test.tmp_assure_order_nocancel
SELECT 
    o.system_source_cd,
    @etl_date,
    o.sjb_ptflag,
    o.order_number,
    o.driver,
    o.driver_name,
    o.company_code,
    o.pay_by_intermediary,
    o.sign_upload_time,
    o.sign_in_state,
    o.dynamic_state,
    o.create_date,
    o.agent_code,
    CASE WHEN o.sign_upload_time IS NOT NULL THEN DATEDIFF(o.sign_upload_time,o.create_date) 
        ELSE NULL 
    END AS trans_days
FROM odm.fs_plt_assure_orders o
INNER JOIN odm.fs_plt_order_stock stock
ON o.order_number=stock.order_number 
    AND stock.sjb_etl_date=@etl_date
    AND stock.stock_kind_name='煤炭'
WHERE o.sjb_etl_date=@etl_date
    AND (o.sign_in_state =8 OR o.dynamic_state NOT IN (2,4,5,10,12,15,16,17)) 
    AND DATE_FORMAT(o.create_date,'%Y')>1970
    AND NOT EXISTS
    (
        select 1
        from repm.test_company_cd testc
        where o.company_code=testc.company_code
     )
;
/*添加索引提升查询效率*/
ALTER TABLE test.tmp_assure_order_nocancel ADD INDEX idx_order_number(order_number);

/*重跑聚类rfm_login样本数据*/
DELETE FROM repm.rep_op_dri_order_and_login_cluster_samples;
INSERT INTO repm.rep_op_dri_order_and_login_cluster_samples 
SELECT 
    @etl_date,
    dri_info.driver_code, -- 司机编码
    (YEAR(@etl_date)-SUBSTR(dri_info.driver_idcard,7,4)) AS dri_age, -- 司机年龄
    orde.com_cnt,-- 服务公司数量
    orde.upload_province,-- 服务卸货省份数量
    orde.loadupload_region,-- 县去重统计
    orde.order_cnt, -- 跑单数量
    orde.agent_order_rate, -- 经纪人运单占比
    dri_freq.order_freq,-- 跑单时间间隔
    orde.per_order_pay,-- 单均金额
    orde.order_pay,-- 下单金额
    orde.per_order_lc,-- 单均里程
    orde.first_order_days, -- 首单距今时长
    orde.last_order_time,-- 最近下单时间
    orde.recent_order_days,-- 最近单距今时长
    login.first_login_days,-- 首次登录距今时长
    login.recent_login_days,-- 最近登录距今时长
    login.login_times, -- 总共登录次数
    login.login_freq
FROM 
(   
    SELECT 
        o.driver,
        COUNT(DISTINCT o.company_code) AS com_cnt,
        COUNT(DISTINCT line.upload_province) AS upload_province,
        COUNT(DISTINCT CONCAT(load_region_name,'_',upload_region_name)) AS loadupload_region,--  县-县去重统计
        COUNT(DISTINCT o.order_number) AS order_cnt,
        COUNT(o.agent_code)/COUNT(DISTINCT o.order_number) AS agent_order_rate,
        SUM(pay_by_intermediary/10000) AS order_pay,
        SUM(pay_by_intermediary/10000)/COUNT(DISTINCT o.order_number) AS per_order_pay,
        AVG(lc.lc) AS per_order_lc,
        MIN(o.create_date) AS first_order_date,
        (DATEDIFF(@etl_date,MIN(o.create_date))+1) AS first_order_days,
        max(o.create_date) as last_order_time,
        (DATEDIFF(@etl_date,MAX(o.create_date))+1) AS recent_order_days
    FROM test.tmp_assure_order_nocancel o
    JOIN repm.rep_op_ord_p_line_info line
    ON o.order_number=line.order_number
    JOIN 
    (
        -- company_code,load_address_detail,upload_address_detail  不是联合主键
        SELECT 
            company_code,
            load_address_detail,
            upload_address_detail,
            AVG(lc) AS lc -- 单位：米
        FROM repm.order_address_lc
        GROUP BY company_code,load_address_detail,upload_address_detail
    ) lc  
    ON line.company_code=lc.company_code AND line.load_address_detail=lc.load_address_detail AND line.upload_address_detail=lc.upload_address_detail
    GROUP BY o.driver
) orde
LEFT JOIN repm.rep_op_dri_p_base_info dri_info
ON dri_info.driver_code=orde.driver
    AND dri_info.driver_idcard IS NOT NULL 
LEFT JOIN 
(   -- 已签收
    SELECT 
        driver,
        -- 下单频率：下次下单距离本次签收平均天数
        (DATEDIFF(MAX(sign_upload_time),MIN(create_date))-SUM(trans_days))/(COUNT(DISTINCT order_number)-1) AS order_freq
    from test.tmp_assure_order_nocancel
    WHERE sign_upload_time IS NOT NULL  AND DATE_FORMAT(sign_upload_time,'%Y')>1970
    GROUP BY driver
) dri_freq
ON orde.driver=dri_freq.driver
LEFT JOIN 
(
    SELECT user_code,
        (DATEDIFF(@etl_date,MIN(sjb_etl_date))+1) AS first_login_days,
        (DATEDIFF(@etl_date,MAX(sjb_etl_date))+1) AS recent_login_days,
        COUNT(DISTINCT sjb_etl_date) AS login_times,
        -- 登录频率
        (DATEDIFF(MAX(sjb_etl_date),MIN(sjb_etl_date)))/(COUNT(DISTINCT sjb_etl_date)-1) AS login_freq
    FROM odm.fs_log_user_login login
    WHERE DATE_FORMAT(sjb_etl_date,'%Y')>1970 AND sjb_etl_date<=@etl_date
    GROUP BY user_code
) login 
ON orde.driver=login.user_code
;

/*记录结束时间*/
SET END_TIME = NOW();

IF msg IS NULL THEN
SELECT 'SUCCESS';
/*写入存储过程日志表中*/
INSERT INTO repm.repm_sp_log (
    SJB_ETL_DATE,
    SP_NAME,
    START_TIME,
    END_TIME,
    STATE,
    LOG_DESC
)
VALUES
    (
        @ETL_DATE,
        'rep_op_dri_order_and_login_cluster_samples',
        START_TIME,
        END_TIME,
        '1',
        'rep_op_dri_order_and_login_cluster_samples报表数据加工成功'
    );
ELSE 
SET @CODE_DESC=concat('rep_op_dri_order_and_login_cluster_samples报表数据加工失败，失败码为',concat(CODE, msg));
SELECT concat('ERROR',concat(CODE, msg));
INSERT INTO repm_SP_LOG (
    SJB_ETL_DATE,
    SP_NAME,
    START_TIME,
    END_TIME,
    STATE,
    LOG_DESC
)
VALUES
    (
        @ETL_DATE,
        'rep_op_dri_order_and_login_cluster_samples',
        START_TIME,
        END_TIME,
        '0',
        @CODE_DESC
    );

END IF;

END
;