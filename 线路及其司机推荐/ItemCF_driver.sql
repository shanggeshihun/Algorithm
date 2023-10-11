˾�� ��· �ϵ��ܸ���·���ʱ��  ж����װ��ƽ����� 


��· ��ʼ�� �ܵ�˾�� 

��·���ƶȣ�
��· ��·��ʼ�� ��·��ʼ���� ����׶��ر���� ��·����˼� ��·����˼�

˾�����ƶȣ�
˾�� ˾������ ˾������ ˾������ܵ���ʼ���� ��·��ʼ���� �ϴ��ܵ�Ƶ��F(�ϴ�ж��-����װ��) �ƻ��ܵ��վ������(now-(�ϴ�ж��+F)) ƽ���˼�


u=line
i=dri

��·�����ܵ���˾��
train={line1:{dri1:,dri2:,dri3:},line2:{}}


˾�����ƶȣ�
W=W[dir1][dri2]


ItemCF(u=line,i=dri)

sample1:
�ܵ�������
line1 dri1 times11
line1 dri2 times12
line2 dri1 times21
line2 dri2 times22  

sample2
����: Ȩ��1,1,1,1,0.5
�߼������������ͬ���1����0����pred_actual_err_days���10�������0����1��
Ŀ�ģ�����˾������ϵ��
bug��Ȩ�����ȷ��
dri1:most_load_city most_load_region most_upload_city most_upload_region most_line recent_line recent_gap_hours


/*=======ÿ����·ÿ��˾���ܵ�����======*/

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



/*===========˾������=========*/
-- ��Ƶ����װ����
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
    where load_province is not null -- �����˵��ķ���ʡ��Ϊ��
    group by driver_code,load_city
    order by driver_code,count(distinct order_number) desc 
    ) pro,
    (select @dri:=null,@rk:=0) r
) p
where rk=1;
alter table test.tmp_most_load_city add index idx_dri(driver_code);

-- ��Ƶ����װ����
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
    where load_province is not null -- �����˵��ķ���ʡ��Ϊ��
    group by driver_code,load_region_name
    order by driver_code,count(distinct order_number) desc 
    ) pro,
    (select @dri:=null,@rk:=0) r
) p
where rk=1;
alter table test.tmp_most_load_region add index idx_dri(driver_code);


-- ��Ƶ����ж����
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
    where upload_province is not null -- �����˵��ķ���ʡ��Ϊ��
    group by driver_code,upload_city
    order by driver_code,count(distinct order_number) desc 
    ) pro,
    (select @dri:=null,@rk:=0) r
) p
where rk=1;
alter table test.tmp_most_upload_city add index idx_dri(driver_code);

-- ��Ƶ����ж����
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
    where upload_province is not null -- �����˵��ķ���ʡ��Ϊ��
    group by driver_code,upload_region_name
    order by driver_code,count(distinct order_number) desc 
    ) pro,
    (select @dri:=null,@rk:=0) r
) p
where rk=1;
alter table test.tmp_most_upload_region add index idx_dri(driver_code);


-- ��Ƶ������-��
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
    where load_region_name is not null and upload_region_name is not null -- �����˵��ķ���ʡ��Ϊ��
    group by driver_code,concat(load_region_name,'_',upload_region_name)
    order by driver_code,count(distinct order_number) desc 
    ) pro,
    (select @dri:=null,@rk:=0) r
) p
where rk=1;
alter table test.tmp_most_region_region add index idx_dri(driver_code);

-- ��Ƶ����line
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
    where load_address is not null and upload_address is not null -- �����˵��ķ���ʡ��Ϊ��
    group by driver_code,concat(load_address,'_',upload_address)
    order by driver_code,count(distinct order_number) desc 
    ) pro,
    (select @dri:=null,@rk:=0) r
) p
where rk=1;
alter table test.tmp_most_line add index idx_dri(driver_code);


-- �������-��
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
    where load_region_name is not null and upload_region_name is not null -- �����˵��ķ���ʡ��Ϊ��
    order by driver_code,order_number desc 
    ) pro,
    (select @dri:=null,@rk:=0) r
) p
where rk=1;
alter table test.tmp_recent_line add index idx_dri(driver_code);


-- ����ܵ�Ƶ��(�������������һ��)
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
    where this_z_province is not null and gap_hours is not null-- �����˵��ķ���ʡ��Ϊ��
    order by driver_code,this_order_number desc
    ) pro,
    (select @dri:=null,@rk:=0) r
) p
where rk=1;
alter table test.tmp_recent_gap_hours add index idx_dri(driver_code);


-- ����˾��ָ��
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
 






������ �����
