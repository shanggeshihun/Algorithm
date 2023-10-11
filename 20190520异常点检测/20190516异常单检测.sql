在无明显业务指向下，从算法分析的角度通过选取运单的各类维度，找出偏离正常运单范围的数据，以此来推动异常运单的监控

运单号 发布货源吨数 实际卸货吨数  (吨数误差比例) 装货地 卸货地 (装卸距离)装货时间 卸货时间 (装卸天数) (时速)发货运价 收单运价 (运价误差比例)




装货地 卸货地 (装卸距离) 运单号 发布货源吨数 实际卸货吨数 (吨数误差比例)  (时速) 发货运价 收单运价 (运价误差比例)  (中位数时速误差)

set @etl_date='2019-07-23';
drop table if exists test.tmp_sign_order_infomation;
-- 临时表 实际装卸货地址之间的距离以及装货到卸货签到耗时
create table test.tmp_sign_order_infomation as 
select 
    s1.order_number,
    s1.longitude as z_longitude,
    s1.latitude as z_latitude,
    s1.address as z_address,
    o.load_city,
    s1.sign_date as z_sign_date,
    s2.longitude as x_longitude,
    s2.latitude as x_latitude,
    s2.address as x_address,
    o.upload_city,
    s2.sign_date as x_sign_date,
        timestampdiff(minute,s1.sign_date,s2.sign_date)/60 as gap_hours,
    round(6378.138*2*asin(sqrt(pow(sin((s1.latitude*pi()/180-s2.latitude*pi()/180)/2),2)+cos(s1.latitude*pi()/180)*cos(s2.latitude*pi()/180)*pow(sin( (s2.longitude*pi()/180-s1.longitude*pi()/180)/2),2)))*1000)/1000 as distance,
    round(6378.138*2*asin(sqrt(pow(sin((s1.latitude*pi()/180-s2.latitude*pi()/180)/2),2)+cos(s1.latitude*pi()/180)*cos(s2.latitude*pi()/180)*pow(sin( (s2.longitude*pi()/180-s1.longitude*pi()/180)/2),2)))*1000)/1000/(timestampdiff(minute,s1.sign_date,s2.sign_date)/60) as speed
from odm.fs_plt_sign_in_order s1
join odm.fs_plt_sign_in_order s2
on s1.order_number=s2.order_number and s2.sjb_etl_date=@etl_date and s2.sign_in_kind=2 -- 卸货地签到
join repm.rep_op_ord_p_line_info o 
on s1.order_number=o.order_number and o.company_code not in (select company_code from repm.test_company_cd)
where s1.sjb_etl_date=@etl_date 
and s1.longitude>0 
and s2.longitude>0
and s1.create_time>='2019-01-01' 
and s1.sign_in_kind=1 -- 装货地签到
order by s1.order_number,s1.sign_date asc;

alter table test.tmp_sign_order_infomation add index idx_order(order_number);

-- 卸货吨数误差、收单运价、收单运价误差、距离、时速
set @etl_date='2019-07-23';
select
    i.load_city,
    i.upload_city,
    r.order_number,
    r.original_unit,
    r.actual_unit,
    if(r.actual_unit/r.original_unit is null,0,abs(r.actual_unit/r.original_unit-1)) as unit_err_rate,
    replace(replace(r.original_unit_price,"元/吨",""),"元/方","") as original_unit_price,
    replace(replace(r.actual_unit_price,"元/吨",""),"元/方","") as actual_unit_price,
    abs(replace(replace(r.actual_unit_price,"元/吨",""),"元/方","")/replace(replace(r.original_unit_price,"元/吨",""),"元/方","")-1) as unit_price_err_rate,
    i.gap_hours,
    i.distance,
    i.speed
from odm.fs_plt_order_sign_received r
join test.tmp_sign_order_infomation i 
on r.order_number=i.order_number
where r.sjb_etl_date=@etl_date 
and r.actual_unit is not null
and r.unit is not null 
order by i.load_city,i.upload_city;
