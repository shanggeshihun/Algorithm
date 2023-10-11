司机 线路 上单跑该线路距今时长  卸货到装货平均间隔 


线路 起始地 跑单司机 

线路相似度：
线路 线路起始地 线路起始地县 相对首都地表距离 线路最高运价 线路最低运价

司机相似度：
司机 司机年龄 司机籍贯 司机最近跑单起始城市 线路起始地县 上次跑单频率F(上次卸货-本次装货) 计划跑单日绝对误差(now-(上次卸货+F)) 平均运价


u=line
i=dri

线路及其跑单的司机
train={line1:{dri1:,dri2:,dri3:},line2:{}}


司机相似度：
W=W[dir1][dri2]


ItemCF(u=line,i=dri)

sample1:
跑单次数：
line1 dri1 times11
line1 dri2 times12
line2 dri1 times21
line2 dri2 times22  

sample2
属性: 权重1,1,1,1,0.5
逻辑：如果否则相同则记1否则0，另，pred_actual_err_days相差10天则记作0否则1；
目的：生成司机相似系数
bug：权重如何确定
dri1:most_load_city most_load_region most_upload_city most_upload_region most_line recent_line recent_gap_hours


/*=======每条线路每个司机跑单数量======*/

create table repm.rep_op_line_order_cnt 
(line varchar(100),load_city varchar(100),load_region_name varchar(100),upload_city varchar(100),upload_region_name varchar(100),driver_code varchar(100),order_cnt int);

insert into repm.rep_op_line_order_cnt 
select 
    concat(load_address,'_',upload_address) as line,
    load_city,
    load_region_name,
    upload_city,
    upload_region_name,
    driver_code,
    count(distinct order_number) as order_cnt
from repm.rep_op_ord_p_line_info
group by 
concat(load_address,'_',upload_address) ,load_city,load_region_name,upload_city,upload_region_name,driver_code
;



/*===========司机属性=========*/
-- 最频繁的装货市
drop table if exists test.tmp_most_load_city;
create table test.tmp_most_load_city as 
select driver_code,load_city as most_load_city
from 
(
    select pro.driver_code,load_city,if(@dri=driver_code,@rk:=@rk+1,@rk:=1) as rk,@dri:=driver_code as v
    from 
    (
    select 
            driver_code,
            load_city,
            count(distinct order_number) as order_cnt
    from repm.rep_op_ord_p_line_info
    where load_province is not null -- 存在运单的发货省份为空
    group by driver_code,load_city
    order by driver_code,count(distinct order_number) desc 
    ) pro,
    (select @dri:=null,@rk:=0) r
) p
where rk=1;
alter table test.tmp_most_load_city add index idx_dri(driver_code);

-- 最频繁的装货区
drop table if exists test.tmp_most_load_region;
create table test.tmp_most_load_region as 
select driver_code,load_region_name as most_load_region
from 
(
    select pro.driver_code,load_region_name,if(@dri=driver_code,@rk:=@rk+1,@rk:=1) as rk,@dri:=driver_code as v
    from 
    (
    select 
            driver_code,
            load_region_name,
            count(distinct order_number) as order_cnt
    from repm.rep_op_ord_p_line_info
    where load_province is not null -- 存在运单的发货省份为空
    group by driver_code,load_region_name
    order by driver_code,count(distinct order_number) desc 
    ) pro,
    (select @dri:=null,@rk:=0) r
) p
where rk=1;
alter table test.tmp_most_load_region add index idx_dri(driver_code);


-- 最频繁的卸货市
drop table if exists test.tmp_most_upload_city;
create table test.tmp_most_upload_city as 
select driver_code,upload_city as most_upload_city
from 
(
    select pro.driver_code,upload_city,if(@dri=driver_code,@rk:=@rk+1,@rk:=1) as rk,@dri:=driver_code as v
    from 
    (
    select 
            driver_code,
            upload_city,
            count(distinct order_number) as order_cnt
    from repm.rep_op_ord_p_line_info
    where upload_province is not null -- 存在运单的发货省份为空
    group by driver_code,upload_city
    order by driver_code,count(distinct order_number) desc 
    ) pro,
    (select @dri:=null,@rk:=0) r
) p
where rk=1;
alter table test.tmp_most_upload_city add index idx_dri(driver_code);

-- 最频繁的卸货区
drop table if exists test.tmp_most_upload_region;
create table test.tmp_most_upload_region as 
select driver_code,upload_region_name as most_upload_region
from 
(
    select pro.driver_code,upload_region_name,if(@dri=driver_code,@rk:=@rk+1,@rk:=1) as rk,@dri:=driver_code as v
    from 
    (
    select 
            driver_code,
            upload_region_name,
            count(distinct order_number) as order_cnt
    from repm.rep_op_ord_p_line_info
    where upload_province is not null -- 存在运单的发货省份为空
    group by driver_code,upload_region_name
    order by driver_code,count(distinct order_number) desc 
    ) pro,
    (select @dri:=null,@rk:=0) r
) p
where rk=1;
alter table test.tmp_most_upload_region add index idx_dri(driver_code);


-- 最频繁的区-区
drop table if exists test.tmp_most_region_region;
create table test.tmp_most_region_region as 
select driver_code,region_region as most_region_region
from 
(
    select pro.driver_code,region_region,if(@dri=driver_code,@rk:=@rk+1,@rk:=1) as rk,@dri:=driver_code as v
    from 
    (
    select 
            driver_code,
            concat(load_region_name,'_',upload_region_name) as region_region,
            count(distinct order_number) as order_cnt
    from repm.rep_op_ord_p_line_info
    where load_region_name is not null and upload_region_name is not null -- 存在运单的发货省份为空
    group by driver_code,concat(load_region_name,'_',upload_region_name)
    order by driver_code,count(distinct order_number) desc 
    ) pro,
    (select @dri:=null,@rk:=0) r
) p
where rk=1;
alter table test.tmp_most_region_region add index idx_dri(driver_code);

-- 最频繁的line
drop table if exists test.tmp_most_line;
create table test.tmp_most_line as 
select driver_code,line as most_line
from 
(
    select pro.driver_code,line,if(@dri=driver_code,@rk:=@rk+1,@rk:=1) as rk,@dri:=driver_code as v
    from 
    (
    select 
            driver_code,
            concat(load_address,'_',upload_address) as line,
            count(distinct order_number) as order_cnt
    from repm.rep_op_ord_p_line_info
    where load_address is not null and upload_address is not null -- 存在运单的发货省份为空
    group by driver_code,concat(load_address,'_',upload_address)
    order by driver_code,count(distinct order_number) desc 
    ) pro,
    (select @dri:=null,@rk:=0) r
) p
where rk=1;
alter table test.tmp_most_line add index idx_dri(driver_code);


-- 最近单区-区
drop table if exists test.tmp_recent_line;
create table test.tmp_recent_line as 
select driver_code,line as recent_line
from 
(
    select pro.driver_code,line,if(@dri=driver_code,@rk:=@rk+1,@rk:=1) as rk,@dri:=driver_code as v
    from 
    (
    select 
        driver_code,
        concat(load_address,'_',upload_address) as line
    from repm.rep_op_ord_p_line_info
    where load_region_name is not null and upload_region_name is not null -- 存在运单的发货省份为空
    order by driver_code,order_number desc 
    ) pro,
    (select @dri:=null,@rk:=0) r
) p
where rk=1;
alter table test.tmp_recent_line add index idx_dri(driver_code);


-- 最近跑单频率(间隔多少天跑下一单)
drop table if exists test.tmp_recent_gap_hours;
create table test.tmp_recent_gap_hours as 
select driver_code,gap_hours as recent_gap_hours,timestampdiff(hour,this_z_sign_date,now())
from 
(
    select pro.*,if(@dri=driver_code,@rk:=@rk+1,@rk:=1) as rk,@dri:=driver_code as v
    from 
    (
    select 
          *
    from repm.rep_op_dri_upload_load_gap
    where this_z_province is not null and gap_hours is not null-- 存在运单的发货省份为空
    order by driver_code,this_order_number desc
    ) pro,
    (select @dri:=null,@rk:=0) r
) p
where rk=1;
alter table test.tmp_recent_gap_hours add index idx_dri(driver_code);


-- 汇总司机指标
drop table if exists repm.rep_op_dri_features;
create table repm.rep_op_dri_features
(
driver_code varchar(50),
most_load_city varchar(100),
most_load_region varchar(100),
most_upload_city varchar(100),
most_upload_region varchar(100),
most_region_region varchar(100),
most_line varchar(100),
recent_line varchar(100),
recent_gap_hours double
);
insert into repm.rep_op_dri_features
select 
    c.driver_code,
    c.most_load_city,
    lr.most_load_region,
    uc.most_upload_city,
    ur.most_upload_region,
    rr.most_region_region,
    ml.most_line,
    rl.recent_line,
    gh.recent_gap_hours
from test.tmp_most_load_city c 
left join test.tmp_most_load_region lr 
on c.driver_code=lr.driver_code
left join test.tmp_most_upload_city uc 
on c.driver_code=uc.driver_code
left join test.tmp_most_upload_region ur
on c.driver_code=ur.driver_code
left join test.tmp_most_line ml 
on c.driver_code=ml.driver_code
left join test.tmp_most_region_region rr
on c.driver_code=rr.driver_code
left join test.tmp_recent_line rl
on c.driver_code=rl.driver_code
left join test.tmp_recent_gap_hours gh
on c.driver_code=gh.driver_code;
 






榆林市 天津市
