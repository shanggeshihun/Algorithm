-- 煤炭 跑单司机聚类结果表
select
    clu.driver_code,
    clu.age_labels_define,
    clu.loss_labels_define,
    clu.active_define_final,
    clu.volumn_define_final,
    clu.lg_labels_define,
    clu.sw_labels_define,
    clu.reply_ag_labels_define,
    clu.com_cnt,
    clu.order_cnt,
    clu.per_order_pay,
    clu.recent_order_days,
    clu.dri_age,
    clu.order_freq,
    clu.order_pay,
    clu.loadupload_region,
    clu.login_freq,
    clu.agent_order_rate,
    clu.search_days,
    case when dinfo.sex=1 then '男' when dinfo.sex=0 then '女'end as sex, 
    case when dinfo.certify_state_all=1 then '未认证'
        when dinfo.certify_state_all=2 then '等待认证'
        when dinfo.certify_state_all=3 then '审核未通过'
        when dinfo.certify_state_all=4 then '认证通过'
        when dinfo.certify_state_all=5 then '身份证认证中'
    end as certify_state,
    dinfo.exc_order_num,
    dinfo.cancel_order_num,
    dinfo.company_name,
    dinfo.load_address,
    dinfo.upload_address,
    dinfo.address
from repm.rep_op_driver_clusters_result clu
left join ldm.t01_f_driver_info dinfo
on clu.driver_code=dinfo.driver_code



用户：plt_user
跑单：公司和司机  plt_assr