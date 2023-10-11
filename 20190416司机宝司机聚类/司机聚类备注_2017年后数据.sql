
1 日志相关表：
    fs_plt_user-用户
    fs_log_user_login-登录  
    fs_log_user_stock_watch-日志解析用户货源查看信息 
    
    fs_log_user_stock_search-日志解析用户货源搜索信息
        search_begin/search_end/search_time/stock_kind
    
2 业务相关表：
    fs_plt_assure_orders-运单信息
    rep_op_ord_p_line_info-运单线路
    order_address_lc-线路里程
    运单、跑单时间、线路
3 资料：
    rep_op_dri_p_base_info-司机注册信息表 
    司机年龄
4 审核：
    rep_op_dri_p_base_info-司机注册信息表
    资料完整度


-- 维度 & 基础数据 20190218
基础数据：17年跑单，货源类型是'煤炭'


1 司机是否流失
指标：'first_order_days','recent_order_days'
(
example：
0 首次跑单4个月 最近跑单距今2月左右 --62.4%  》 潜力跑单客户
1 首次跑单19个月 最近跑单距今17月左右 -- 5.5%  》 流失客户
2 首次跑单10个月 最近跑单距今6月左右 -- 31.9%  》 潜在流失客户
)

2 司机是否活跃
指标：'recent_order_days','order_freq'
**频率：(最近跑单日-最初跑单日)/跑单数量
(
example：
0 最近跑单2.3个月 频率9天 -- 74.4%  》 较活跃
1 最近跑单11个月 频率4天 -- 19.2%  》 不活跃
2 最近跑单2.6个月 频率90天 -- 6.4%  》 低活跃
)


3 司机体量大小
指标：'order_cnt','per_order_pay','order_pay'
(
example：
0 跑单8单 下单金额3.6万 单均0.46万--62.3%  》 一般体量
1 跑单5单 下单金额5.3万 单均1.1万--33.3%  》 高单均
2 跑单86单下单金额38.6万            4.35%  》 大体量
)

4 APP粘性：
指标：'recent_login_days','login_times','login_freq'
17年后 最近登录距今、登录次数、平均登录间隔
(
example：
0 最近登录距今2.7月左右 间隔2个月 -- 9.2% 》 APP粘性较低
1 最近登录距今2.3月左右 间隔6天天--61.0% 》 APP粘性有潜力提升 
3 最近登录距今8月左右 -- 29.6% 》 APP粘性低(最低)
)

5 货源意向：
17年后 首次搜索距今、最近搜索距今、搜索天次数、平均搜索间隔
指标：'search_days','first_search_days','recent_search_days','watch_days','first_watch_days','recent_watch_days','search_freq','watch_freq'
(
example：
3 最近搜索货源2.3个月 最近查看货源1个月  --56.7%  查看行为比较新、查看次数低》 有意向获取货源，潜在引导型

1 最近搜索货源8个月前 最近查看货源9个月 平均搜索5天次   --15.8%  长时间无搜索行为》 低意向获取货源，可能流失型

0 最近搜索9个月 最近查看货源0个月                    -- 25.4%  很早有搜索行为 近来有查看》 低意向获取货源，可能流失型

2 最近搜索1.5个月 最近查看货源2.5个月 查看搜索次数高  -  2.0%   最近查看比较近》 高意向获取货源
)

6 货物来源经纪人依赖：
agent_order_cnt_rate(70% + 强依赖，50%+较强依赖，20%+一般依赖，0轻依赖，=0不依赖)

7 服务多样性：
指标：'com_cnt','upload_province'
(
example：
0 单个公司单个省份  -- 80.0%  》 公司及卸货省单一
1 多个公司多个省份  -- 9.0%  》 公司及卸货省多样
2 多个公司单个省份  -- 10.9%  》 公司多样卸货省单一
)

8 跑单线路多样性：



fs_plt_user_membership_log  用户会员分值流水表

-- 创建临时有效运单表（剔除取消订单及测试公司，煤炭类型，17以后）
drop table if exists repm.assure_orders_temp;
set @etl_date='2019-02-20';
create table repm.assure_orders_temp as 
select o.* ,
    -- 签收单则记录运输耗费天数
	case when o.sign_upload_time is not null then datediff(o.sign_upload_time,o.create_date) else null end as trans_days
from odm.fs_plt_assure_orders o
inner join odm.fs_plt_order_stock stock
on o.order_number=stock.order_number 
    and stock.sjb_etl_date=@etl_date
    and stock.stock_kind_name='煤炭'
where o.sjb_etl_date=@etl_date
    and (o.sign_in_state =8 OR o.dynamic_state NOT in (2,4,5,10,12,15,16,17)) 
    and date_format(o.create_date,'%Y')>='2017'
    and not exists
    (
        select 1
        from repm.test_company_cd testc
        where o.company_code=testc.company_code
     )
;


-- 司机跑单流失情况  司机平台登录情况  司机体量  司机跑单多样性
set @etl_date='2019-02-20';
SELECT 
	dri_info.driver_code, -- 司机编码
	(YEAR(@etl_date)-substr(dri_info.driver_idcard,7,4)) as dri_age, -- 司机年龄
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
    orde.recent_order_days,-- 最近单距今时长
	login.first_login_days,-- 首次登录距今时长
    login.recent_login_days,-- 最近登录距今时长
    login.login_times, -- 总共登录次数
    login.login_freq 
FROM 
(   
    Select 
        o.driver,
        count(distinct o.company_code) as com_cnt,
        count(distinct line.upload_province) as upload_province,
        count(distinct concat(load_region_name,'_',upload_region_name)) as loadupload_region,--  县-县去重统计
        count(distinct o.order_number) as order_cnt,
        count(o.agent_code)/count(distinct o.order_number) as agent_order_rate,
        sum(pay_by_intermediary/10000) as order_pay,
        sum(pay_by_intermediary/10000)/count(distinct o.order_number) as per_order_pay,
        avg(lc.lc) as per_order_lc,
        min(o.create_date) as first_order_date,
        (datediff(@etl_date,min(o.create_date))+1) as first_order_days,
        max(o.create_date) as last_order_time,
        (datediff(@etl_date,max(o.create_date))+1) as recent_order_days
    FROM repm.assure_orders_temp o -- 17年后煤炭 有效运单
    left join repm.rep_op_ord_p_line_info line
    on o.order_number=line.order_number
    left join 
    (
        -- company_code,load_address_detail,upload_address_detail  不是联合主键
        select 
            load_address_detail,
            upload_address_detail,
            avg(lc) as lc -- 单位：米
        from repm.order_address_lc
        group by load_address_detail,upload_address_detail
    ) lc  
    on line.load_address_detail=lc.load_address_detail and line.upload_address_detail=lc.upload_address_detail
    group by o.driver
) orde
left join repm.rep_op_dri_p_base_info dri_info
on dri_info.driver_code=orde.driver
    and dri_info.driver_idcard is not null 
left join 
(   
    select 
        driver,
        (datediff(max(sign_upload_time),min(create_date))-sum(trans_days))/count(distinct order_number) as order_freq
    from repm.assure_orders_temp
    where sign_upload_time is not null and date_format(sign_upload_time,'%Y')>=2017
    group by driver
) dri_freq
on orde.driver=dri_freq.driver
left join 
(
    select user_code,
        (datediff(@etl_date,min(sjb_etl_date))+1) as first_login_days,
        (datediff(@etl_date,max(sjb_etl_date))+1) as recent_login_days,
        count(distinct sjb_etl_date) as login_times,
        (datediff(max(sjb_etl_date),min(sjb_etl_date))+1)/count(distinct sjb_etl_date) as login_freq
    from odm.fs_log_user_login login
    where sjb_etl_date>='2017-01-01' and sjb_etl_date<=@etl_date
    group by user_code
) login 
on orde.driver=login.user_code
;


-- 司机对货源的搜索查看数据
set @etl_date='2019-02-20';
select s.user_code,
	s.search_days,
	s.first_search_days,
	s.recent_search_days,
    s.search_freq,
	w.watch_days,
	w.first_watch_days,
	w.recent_watch_days,
    w.watch_freq
from 
(
    
    select user_code,
        count(distinct sjb_etl_date) as search_days,
        (datediff(@etl_date,min(sjb_etl_date))+1) as first_search_days,
        (datediff(@etl_date,max(sjb_etl_date))+1) as recent_search_days,
        (datediff(max(sjb_etl_date),min(sjb_etl_date))+1)/count(distinct sjb_etl_date) as search_freq
    from odm.fs_log_user_stock_search
    where user_code is not null and date_format(sjb_etl_date,'%Y')>='2017'
        and sjb_etl_date<=@etl_date
    -- 2017年后的search数据
    group by user_code
) s
left join 
(
    
    select user_code,
        count(distinct sjb_etl_date) as watch_days,
        (datediff(@etl_date,min(sjb_etl_date))+1) as first_watch_days,
        (datediff(@etl_date,max(sjb_etl_date))+1) as recent_watch_days,
        (datediff(max(sjb_etl_date),min(sjb_etl_date))+1)/count(distinct sjb_etl_date) as watch_freq
    from odm.fs_log_user_stock_watch
    where user_code is not null and date_format(sjb_etl_date,'%Y')>='2017'
    -- 2017年后的watch数据
        and sjb_etl_date<=@etl_date
    group by user_code
) w
on s.user_code=w.user_code
;
