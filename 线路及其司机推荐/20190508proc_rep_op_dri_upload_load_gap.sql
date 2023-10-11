# 司机上次卸货距离本次装货间隔时长
rep_op_dri_upload_load_gap
DROP PROCEDURE IF EXISTS repm.proc_rep_op_dri_upload_load_gap;
CREATE PROCEDURE repm.proc_rep_op_dri_upload_load_gap(IN ETL_DATE VARCHAR(8))
BEGIN
 /*===============================================================+
         版权信息：版权所有(c) 2017，物易云通
         作业名称：司机卸货装货间隔天数
         责任人  : 
         版本号  : v1.0.0.0
         目标表  : repm.rep_op_dri_upload_load_gap
         备注    :

         修改历史:
         版本     更改日期                      更改人             更改说明
    -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
         v1.0.0.0 2019-04-19 16:20                 liuan                   生成代码
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


-- 临时表记录：司机 运单 装卸货签收信息
drop table if exists test.tmp_driver_sign_order_info;
create table test.tmp_driver_sign_order_info
(
driver_code  varchar(32) DEFAULT NULL COMMENT '接单司机',
order_number  varchar(32) DEFAULT NULL COMMENT '订单号',
company_code varchar(20) DEFAULT NULL COMMENT '企业编码',
company_name varchar(200) DEFAULT NULL COMMENT '企业名称',
z_longitude double DEFAULT NULL COMMENT '装货地经度',
z_latitude double DEFAULT NULL COMMENT '装货地纬度',
z_province varchar(200) DEFAULT NULL COMMENT '装货地省份',
z_city varchar(200) DEFAULT NULL COMMENT '装货地市',
z_region_name varchar(200) DEFAULT NULL COMMENT '装货地区',
z_address varchar(255) DEFAULT NULL COMMENT '装货地址',
z_sign_date datetime DEFAULT NULL COMMENT '装货签到时间',
x_longitude double DEFAULT NULL COMMENT '卸货地经度',
x_latitude double DEFAULT NULL COMMENT '卸货地纬度',
x_province varchar(200) DEFAULT NULL COMMENT '卸货地省份',
x_city varchar(200) DEFAULT NULL COMMENT '卸货地市',
x_region_name varchar(200) DEFAULT NULL COMMENT '卸货地区',
x_address varchar(255) DEFAULT NULL COMMENT '卸货地址',
x_sign_date datetime DEFAULT NULL COMMENT '卸货签到时间'
)
;
insert into test.tmp_driver_sign_order_info 
select 
    s1.driver_code,-- 司机编码
    s1.order_number,
    s1.company_code,
    com.company_name,
    s1.LOAD_longitude as z_longitude,
    s1.LOAD_latitude as z_latitude,
    s1.load_province as z_province,
    s1.load_city as z_city,
    s1.load_region_name as z_region_name,
    s1.load_address as z_address,
    s1.sign_load_time as z_sign_date,
    s1.UPLOAD_longitude as x_longitude,
    s1.UPLOAD_latitude as x_latitude,
    s1.upload_province as x_province,
    s1.upload_city as x_city,
    s1.upload_region_name as x_region_name,
    s1.upload_address as x_address,
    s1.sign_upload_time as x_sign_date
from repm.rep_op_ord_p_line_info s1
join odm.fs_plt_company com 
on s1.company_code=com.company_code and com.sjb_etl_date=@etl_date
where s1.company_code not in (select company_code from repm.test_company_cd)
-- and s1.create_time>='2019-01-01' 
order by s1.driver_code,s1.order_number asc
;

-- 临时表记录：司机编码 跑单序数
drop table if exists test.tmp_driver_sign_order_info_rank;
create table test.tmp_driver_sign_order_info_rank
(
driver_code  varchar(32) DEFAULT NULL COMMENT '接单司机',
order_number  varchar(32) DEFAULT NULL COMMENT '订单号',
company_code varchar(20) DEFAULT NULL COMMENT '企业编码',
company_name varchar(200) DEFAULT NULL COMMENT '企业名称',
z_longitude double DEFAULT NULL COMMENT '装货地经度',
z_latitude double DEFAULT NULL COMMENT '装货地纬度',
z_province varchar(200) DEFAULT NULL,
z_city varchar(200) DEFAULT NULL,
z_region_name varchar(200) DEFAULT NULL,
z_address varchar(255) DEFAULT NULL COMMENT '装货地址',
z_sign_date datetime DEFAULT NULL COMMENT '装货签到时间',
x_longitude double DEFAULT NULL COMMENT '卸货地经度',
x_latitude double DEFAULT NULL COMMENT '卸货地纬度',
x_province varchar(200) DEFAULT NULL,
x_city varchar(200) DEFAULT NULL,
x_region_name varchar(200) DEFAULT NULL,
x_address varchar(255) DEFAULT NULL COMMENT '卸货地址',
x_sign_date datetime DEFAULT NULL COMMENT '卸货签到时间',
dri_order_rank int,
v varchar(100)
)
;

insert into test.tmp_driver_sign_order_info_rank
(driver_code,order_number,company_code,company_name,z_longitude,z_latitude,z_province,z_city,z_region_name,z_address,z_sign_date,x_longitude,x_latitude,x_province,x_city,x_region_name,x_address,x_sign_date,dri_order_rank,v)
select 
    info.driver_code,
    info.order_number,
    info.company_code,
    info.company_name,
    info.z_longitude,
    info.z_latitude,
    info.z_province,
    info.z_city,
    info.z_region_name,
    info.z_address,
    info.z_sign_date,
    info.x_longitude,
    info.x_latitude,
    info.x_province,
    info.x_city,
    info.x_region_name,
    info.x_address,
    info.x_sign_date,
    if(info.driver_code=@dri,@rk:=@rk+1,@rk:=1) as dri_order_rank,
    @dri:=driver_code as v
from test.tmp_driver_sign_order_info info,
    (select @dri:=NULL,@rk:=0) r
;

alter table test.tmp_driver_sign_order_info_rank add index idx_dri_num(driver_code)
;


/*
司机编码  上次运单号 上次卸货地(实际)  上次卸货地时间 本次运单号 本次装货地  本次装货地时间 卸货-装货距离  间隔时长  异常时速(>80KM/h)
drop table if exists repm.rep_op_dri_upload_load_gap;
create table repm.rep_op_dri_upload_load_gap
(
data_dt date,
driver_code  varchar(32) DEFAULT NULL COMMENT '接单司机编码',
last_order_number  varchar(32) NOT NULL COMMENT '上次跑单订单号',
last_x_company_name  varchar(200) NOT NULL COMMENT '上次跑单公司',
last_x_address varchar(255) DEFAULT NULL COMMENT '上次卸货地址',
last_x_province varchar(255) DEFAULT NULL COMMENT '上次卸货省份',
last_x_city varchar(255) DEFAULT NULL COMMENT '上次卸货市',
last_x_region_name varchar(255) DEFAULT NULL COMMENT '上次卸货区',
last_x_sign_date datetime DEFAULT NULL COMMENT '上次卸货签到时间',
this_order_number  varchar(32) DEFAULT NULL COMMENT '本次跑单订单号',
this_z_company_name varchar(200) DEFAULT NULL COMMENT '本次跑单公司',
this_z_address varchar(255) DEFAULT NULL COMMENT '本次装货地址',
this_z_province varchar(255) DEFAULT NULL COMMENT '本次装货省',
this_z_city varchar(255) DEFAULT NULL COMMENT '本次装货市',
this_z_region_name varchar(255) DEFAULT NULL COMMENT '本次装货区',
this_z_sign_date datetime DEFAULT NULL COMMENT '本次装货签到时间',
gap_hours decimal(16,4) DEFAULT NULL COMMENT '上次卸货到本次装货间隔时长(h)',
distance decimal(16,4) DEFAULT NULL COMMENT '上次卸货地到本次装货地距离(km)',
speed decimal(16,2) DEFAULT NULL COMMENT '时速(km/h)'
) COMMENT '按司机统计上次卸货及本次装货对应关系报表';
*/

/*重跑*/
delete from repm.rep_op_dri_upload_load_gap;

/*插入数据*/
insert into repm.rep_op_dri_upload_load_gap
select 
    @etl_date,
    r1.driver_code,
    r1.order_number as last_order_number,r1.company_name as last_x_company_name,r1.x_address as last_x_address,
    r1.x_province as last_x_province,r1.x_city as last_x_city,r1.x_region_name as last_x_region_name, -- r1.x_longitude,r1.x_latitude,
    r1.x_sign_date as last_x_sign_date,
    
    
    r2.order_number as this_order_number,r2.company_name as this_z_company_name,r2.z_address as this_z_address,
    r2.z_province as this_z_province,r2.z_city as this_z_city,r2.z_region_name as this_z_region_name,
    -- r2.z_longitude,r2.z_latitude,
    r2.z_sign_date as this_z_sign_date,

    timestampdiff(minute,r1.x_sign_date,r2.z_sign_date)/60 as gap_hours,
    round(6378.138*2*asin(sqrt(pow(sin((r1.x_latitude*pi()/180-r2.z_latitude*pi()/180)/2),2)+cos(r1.x_latitude*pi()/180)*cos(r2.z_latitude*pi()/180)*pow(sin( (r2.z_longitude*pi()/180-r1.x_longitude*pi()/180)/2),2)))*1000)/1000 as distance,
    round(6378.138*2*asin(sqrt(pow(sin((r1.x_latitude*pi()/180-r2.z_latitude*pi()/180)/2),2)+cos(r1.x_latitude*pi()/180)*cos(r2.z_latitude*pi()/180)*pow(sin( (r2.z_longitude*pi()/180-r1.x_longitude*pi()/180)/2),2)))*1000)/1000/timestampdiff(hour,r1.x_sign_date,r2.z_sign_date) as speed
from test.tmp_driver_sign_order_info_rank r1 
inner join test.tmp_driver_sign_order_info_rank r2
on r1.driver_code=r2.driver_code and r1.dri_order_rank+1=r2.dri_order_rank
where date_format(r1.x_sign_date,'%Y')>1970 and r1.x_latitude>0 and r1.x_latitude>0 and r2.z_longitude>0 and r2.z_latitude>0
-- and round(6378.138*2*asin(sqrt(pow(sin((r1.x_latitude*pi()/180-r2.z_latitude*pi()/180)/2),2)+cos(r1.x_latitude*pi()/180)*cos(r2.z_latitude*pi()/180)*pow(sin( (r2.z_longitude*pi()/180-r1.x_longitude*pi()/180)/2),2)))*1000)/1000/timestampdiff(hour,r1.x_sign_date,r2.z_sign_date)>80
order by r1.order_number asc
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
        'rep_op_dri_upload_load_gap',
        START_TIME,
        END_TIME,
        '1',
        'rep_op_dri_upload_load_gap报表数据加工成功'
    );
ELSE 
SET @CODE_DESC=concat('rep_op_dri_upload_load_gap报表数据加工失败，失败码为',concat(CODE, msg));
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
        'rep_op_dri_upload_load_gap',
        START_TIME,
        END_TIME,
        '0',
        @CODE_DESC
    );

END IF;

END
;

