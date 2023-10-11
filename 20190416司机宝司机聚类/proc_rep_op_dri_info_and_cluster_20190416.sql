
DROP PROCEDURE IF EXISTS repm.proc_rep_op_dri_info_and_cluster;
CREATE PROCEDURE repm.proc_rep_op_dri_info_and_cluster(IN ETL_DATE VARCHAR(8))
BEGIN
 /*===============================================================+
         版权信息：版权所有(c) 2017，物易云通
         作业名称：司机信息及聚类结果报表
         责任人  : 
         版本号  : v1.0.0.0
         目标表  : repm.rep_op_dri_info_and_cluster
         备注    :

         修改历史:
         版本     更改日期                      更改人             更改说明
    -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
         v1.0.0.0 2019-04-16 17:45              liuan              生成代码
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

/*重跑 司机信息及聚类结果报表*/
DELETE FROM repm.rep_op_dri_info_and_cluster;

INSERT INTO repm.rep_op_dri_info_and_cluster
SELECT 
    @ETL_DATE,
    clu_res.driver_code,-- 司机编码
    dri.driver_name,-- 司机姓名
    dri.driver_create_date,-- 司机注册日期
    dri.orders_create_date_min,-- 首次接单时间
    dri.address,-- 户口所在地
    CASE WHEN dri.sex=1 THEN '男' WHEN dri.sex=0 THEN '女' END AS sex,
    case when dri.certify_state_all=1 then '未认证' 
        when dri.certify_state_all=2 then '等待认证'
        when dri.certify_state_all=3 then '审核未通过'
        when dri.certify_state_all=4 then '认证通过'
        when dri.certify_state_all=5 then '身份证认证中'
    end as certify_state_all,
    dri.company_name as common_company_name,-- 常用公司名称
    dri.load_address as common_load_address, -- 常用装货地
    dri.upload_address as common_upload_address,-- 常用卸货地
    dri.exc_order_num,-- 异常单
    dri.cancel_order_num,-- 取消单
    dri.membership_value,-- 积分
    (LENGTH(dri.plate_numbers)-length(replace(dri.plate_numbers,',',''))+1) as plate_numbers_cnt,-- 车牌数量
    clu_res.com_cnt, -- 跑单公司数量
    clu_res.upload_province as upload_province_cnt,-- 卸货省份数量
    clu_res.order_cnt,-- 跑单数量
    clu_res.per_order_lc,-- 单均里程
    clu_res.per_order_pay,-- 单均金额
    clu_res.order_pay,-- 下单总额
    clu_res.volumn_define_final,-- 下单总额标签
    clu_res.recent_order_days,-- 最近单距今天数
    clu_res.loss_labels_define, -- 最近单距今天数标签
    clu_res.dri_age,-- 司机年龄
    clu_res.age_labels_define,-- 司机年龄标签
    clu_res.order_freq,-- 下单间隔天数
    clu_res.active_define_final,-- 下单间隔标签
    clu_res.loadupload_region as loadupload_region_cnt,-- 跑单县-县 条数
    clu_res.sc_labels_define,-- 跑单县-县标签
    clu_res.recent_login_days,-- 最近登录APP距今天数
    clu_res.lg_labels_define,-- APP 粘性标签
    clu_res.recent_search_days,-- 最近搜索货源天数
    clu_res.sw_labels_define,-- 货源意向标签
    clu_res.agent_order_rate,-- 经纪人运单占比
    clu_res.reply_ag_labels_define -- 司机对经纪人依赖程度标签
from repm.rep_op_dri_clusters_result clu_res  -- 聚类结果表
left join ldm.t01_f_driver_info dri  -- 司机信息事实表
on clu_res.driver_code=dri.driver_code
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
        'rep_op_dri_info_and_cluster',
        START_TIME,
        END_TIME,
        '1',
        'rep_op_dri_info_and_cluster报表数据加工成功'
    );
ELSE 
SET @CODE_DESC=concat('rep_op_dri_info_and_cluster报表数据加工失败，失败码为',concat(CODE, msg));
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
        'rep_op_dri_info_and_cluster',
        START_TIME,
        END_TIME,
        '0',
        @CODE_DESC
    );

END IF;
END
;