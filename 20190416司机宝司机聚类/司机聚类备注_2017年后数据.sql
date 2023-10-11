
1 ��־��ر�
    fs_plt_user-�û�
    fs_log_user_login-��¼  
    fs_log_user_stock_watch-��־�����û���Դ�鿴��Ϣ 
    
    fs_log_user_stock_search-��־�����û���Դ������Ϣ
        search_begin/search_end/search_time/stock_kind
    
2 ҵ����ر�
    fs_plt_assure_orders-�˵���Ϣ
    rep_op_ord_p_line_info-�˵���·
    order_address_lc-��·���
    �˵����ܵ�ʱ�䡢��·
3 ���ϣ�
    rep_op_dri_p_base_info-˾��ע����Ϣ�� 
    ˾������
4 ��ˣ�
    rep_op_dri_p_base_info-˾��ע����Ϣ��
    ����������


-- ά�� & �������� 20190218
�������ݣ�17���ܵ�����Դ������'ú̿'


1 ˾���Ƿ���ʧ
ָ�꣺'first_order_days','recent_order_days'
(
example��
0 �״��ܵ�4���� ����ܵ����2������ --62.4%  �� Ǳ���ܵ��ͻ�
1 �״��ܵ�19���� ����ܵ����17������ -- 5.5%  �� ��ʧ�ͻ�
2 �״��ܵ�10���� ����ܵ����6������ -- 31.9%  �� Ǳ����ʧ�ͻ�
)

2 ˾���Ƿ��Ծ
ָ�꣺'recent_order_days','order_freq'
**Ƶ�ʣ�(����ܵ���-����ܵ���)/�ܵ�����
(
example��
0 ����ܵ�2.3���� Ƶ��9�� -- 74.4%  �� �ϻ�Ծ
1 ����ܵ�11���� Ƶ��4�� -- 19.2%  �� ����Ծ
2 ����ܵ�2.6���� Ƶ��90�� -- 6.4%  �� �ͻ�Ծ
)


3 ˾��������С
ָ�꣺'order_cnt','per_order_pay','order_pay'
(
example��
0 �ܵ�8�� �µ����3.6�� ����0.46��--62.3%  �� һ������
1 �ܵ�5�� �µ����5.3�� ����1.1��--33.3%  �� �ߵ���
2 �ܵ�86���µ����38.6��            4.35%  �� ������
)

4 APPճ�ԣ�
ָ�꣺'recent_login_days','login_times','login_freq'
17��� �����¼��񡢵�¼������ƽ����¼���
(
example��
0 �����¼���2.7������ ���2���� -- 9.2% �� APPճ�Խϵ�
1 �����¼���2.3������ ���6����--61.0% �� APPճ����Ǳ������ 
3 �����¼���8������ -- 29.6% �� APPճ�Ե�(���)
)

5 ��Դ����
17��� �״�����������������������������ƽ���������
ָ�꣺'search_days','first_search_days','recent_search_days','watch_days','first_watch_days','recent_watch_days','search_freq','watch_freq'
(
example��
3 ���������Դ2.3���� ����鿴��Դ1����  --56.7%  �鿴��Ϊ�Ƚ��¡��鿴�����͡� �������ȡ��Դ��Ǳ��������

1 ���������Դ8����ǰ ����鿴��Դ9���� ƽ������5���   --15.8%  ��ʱ����������Ϊ�� �������ȡ��Դ��������ʧ��

0 �������9���� ����鿴��Դ0����                    -- 25.4%  ������������Ϊ �����в鿴�� �������ȡ��Դ��������ʧ��

2 �������1.5���� ����鿴��Դ2.5���� �鿴����������  -  2.0%   ����鿴�ȽϽ��� �������ȡ��Դ
)

6 ������Դ������������
agent_order_cnt_rate(70% + ǿ������50%+��ǿ������20%+һ��������0��������=0������)

7 ��������ԣ�
ָ�꣺'com_cnt','upload_province'
(
example��
0 ������˾����ʡ��  -- 80.0%  �� ��˾��ж��ʡ��һ
1 �����˾���ʡ��  -- 9.0%  �� ��˾��ж��ʡ����
2 �����˾����ʡ��  -- 10.9%  �� ��˾����ж��ʡ��һ
)

8 �ܵ���·�����ԣ�



fs_plt_user_membership_log  �û���Ա��ֵ��ˮ��

-- ������ʱ��Ч�˵����޳�ȡ�����������Թ�˾��ú̿���ͣ�17�Ժ�
drop table if exists repm.assure_orders_temp;
set @etl_date='2019-02-20';
create table repm.assure_orders_temp as 
select o.* ,
    -- ǩ�յ����¼����ķ�����
	case when o.sign_upload_time is not null then datediff(o.sign_upload_time,o.create_date) else null end as trans_days
from odm.fs_plt_assure_orders o
inner join odm.fs_plt_order_stock stock
on o.order_number=stock.order_number 
    and stock.sjb_etl_date=@etl_date
    and stock.stock_kind_name='ú̿'
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


-- ˾���ܵ���ʧ���  ˾��ƽ̨��¼���  ˾������  ˾���ܵ�������
set @etl_date='2019-02-20';
SELECT 
	dri_info.driver_code, -- ˾������
	(YEAR(@etl_date)-substr(dri_info.driver_idcard,7,4)) as dri_age, -- ˾������
	orde.com_cnt,-- ����˾����
    orde.upload_province,-- ����ж��ʡ������
    orde.loadupload_region,-- ��ȥ��ͳ��
	orde.order_cnt, -- �ܵ�����
    orde.agent_order_rate, -- �������˵�ռ��
    dri_freq.order_freq,-- �ܵ�ʱ����
    orde.per_order_pay,-- �������
    orde.order_pay,-- �µ����
    orde.per_order_lc,-- �������
    orde.first_order_days, -- �׵����ʱ��
    orde.recent_order_days,-- ��������ʱ��
	login.first_login_days,-- �״ε�¼���ʱ��
    login.recent_login_days,-- �����¼���ʱ��
    login.login_times, -- �ܹ���¼����
    login.login_freq 
FROM 
(   
    Select 
        o.driver,
        count(distinct o.company_code) as com_cnt,
        count(distinct line.upload_province) as upload_province,
        count(distinct concat(load_region_name,'_',upload_region_name)) as loadupload_region,--  ��-��ȥ��ͳ��
        count(distinct o.order_number) as order_cnt,
        count(o.agent_code)/count(distinct o.order_number) as agent_order_rate,
        sum(pay_by_intermediary/10000) as order_pay,
        sum(pay_by_intermediary/10000)/count(distinct o.order_number) as per_order_pay,
        avg(lc.lc) as per_order_lc,
        min(o.create_date) as first_order_date,
        (datediff(@etl_date,min(o.create_date))+1) as first_order_days,
        max(o.create_date) as last_order_time,
        (datediff(@etl_date,max(o.create_date))+1) as recent_order_days
    FROM repm.assure_orders_temp o -- 17���ú̿ ��Ч�˵�
    left join repm.rep_op_ord_p_line_info line
    on o.order_number=line.order_number
    left join 
    (
        -- company_code,load_address_detail,upload_address_detail  ������������
        select 
            load_address_detail,
            upload_address_detail,
            avg(lc) as lc -- ��λ����
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


-- ˾���Ի�Դ�������鿴����
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
    -- 2017����search����
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
    -- 2017����watch����
        and sjb_etl_date<=@etl_date
    group by user_code
) w
on s.user_code=w.user_code
;
