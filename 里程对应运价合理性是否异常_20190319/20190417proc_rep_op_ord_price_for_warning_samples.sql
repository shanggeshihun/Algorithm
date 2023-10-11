
DROP PROCEDURE IF EXISTS repm.proc_rep_op_ord_price_for_warning_samples;
CREATE PROCEDURE repm.proc_rep_op_ord_price_for_warning_samples(IN ETL_DATE VARCHAR(8))
BEGIN
 /*===============================================================+
         ��Ȩ��Ϣ����Ȩ����(c) 2017��������ͨ
         ��ҵ���ƣ���һ���˼��쳣��������
         ������  : 
         �汾��  : v1.0.0.0
         Ŀ���  : repm.rep_op_ord_price_for_warning_samples
         ��ע    :

         �޸���ʷ:
         �汾     ��������                      ������             ����˵��
    -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
         v1.0.0.0 2019-04-17 16:13                 liuan                   ���ɴ���
+===============================================================*/

DECLARE `CODE` CHAR(5) DEFAULT '00000';
DECLARE msg TEXT;
DECLARE START_TIME DATETIME;
DECLARE END_TIME DATETIME;
DECLARE CONTINUE HANDLER FOR SQLEXCEPTION
    BEGIN
        /*ͨ��GET DIAGNOSTICS�����ķ�ʽ��ȡsqlstate message_text(��Ȼ������Ϣ�����ſ����Լ�����)*/
        GET DIAGNOSTICS CONDITION 1 CODE = RETURNED_SQLSTATE,
        msg = MESSAGE_TEXT;
    END;
SET @ETL_DATE = STR_TO_DATE(ETL_DATE,'%Y%m%d');
/*����ʱ��*/
SET START_TIME = NOW();

/*����*/
DELETE FROM repm.rep_op_ord_price_for_warning_samples;
INSERT repm.rep_op_ord_price_for_warning_samples
(
 data_dt
,company_check_receipt_date
,company_code
,company_name
,order_number
,load_address
,upload_address
,lc
,estimate_price
,actual_unit_price
,order_create_time
,driver_code
,driver_name
,driver_phone
)
SELECT *
FROM 
(
    SELECT 
        @ETL_DATE as data_dt,
        DATE_FORMAT(a.company_check_receipt_time,'%Y-%m-%d') as company_check_receipt_date
        ,a.company_code
        ,b.company_name
        ,a.order_number
        ,a.load_address
        ,a.upload_address
        ,b.lc
        ,19.368612570820474 +0.26126693*b.LC/1000  as estimate_price
        ,REPLACE(c.actual_unit_price,'Ԫ/��','') as actual_unit_price
        ,a.order_create_time as order_create_time
        ,e.direct_driver_code as driver_code
        ,e.direct_driver_name as driver_name
        ,e.direct_driver_phone as driver_phone
        FROM repm.rep_op_ord_p_line_info as a
        INNER JOIN 
        (
            SELECT company_code,company_name,load_adress,load_address_detail,upload_adress,upload_address_detail,MIN(LC) as lc
            FROM order_address_lc
            GROUP BY company_code,company_name,load_adress,load_address_detail,upload_adress,upload_address_detail
        ) as b
        ON a.company_code=b.company_code AND a.load_address=b.load_adress AND a.load_address_detail=b.load_address_detail AND a.upload_address=b.upload_adress AND a.upload_address_detail=b.upload_address_detail
        INNER JOIN odm.fs_plt_order_sign_received as c
        ON a.order_number=c.order_number AND c.sjb_etl_date=@ETL_DATE
        INNER JOIN odm.fs_plt_order_stock as d
        ON a.order_number=d.order_number AND d.sjb_etl_date=@ETL_DATE
        INNER JOIN odm.fs_plt_order_driver as e
        ON a.order_number=e.order_number AND e.sjb_etl_date=@ETL_DATE
        /*��һ���ȷ�ϻص�����*/
        WHERE DATE_FORMAT(a.company_check_receipt_time,'%Y-%m-%d')<=@ETL_DATE AND DATE_FORMAT(a.company_check_receipt_time,'%Y-%m-%d')>=DATE_SUB(@ETL_DATE,INTERVAL 365 day)
        AND d.stock_kind_name='ú̿'
        AND INSTR(actual_unit_price,'Ԫ/��')>0
        AND a.company_code NOT IN (SELECT company_code FROM repm.test_company_cd)
)  a
;


/*��¼����ʱ��*/
SET END_TIME = NOW();

IF msg IS NULL THEN
SELECT 'SUCCESS';
/*д��洢������־����*/
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
        'rep_op_ord_price_for_warning_samples',
        START_TIME,
        END_TIME,
        '1',
        'rep_op_ord_price_for_warning_samples�������ݼӹ��ɹ�'
    );
ELSE 
SET @CODE_DESC=concat('rep_op_ord_price_for_warning_samples�������ݼӹ�ʧ�ܣ�ʧ����Ϊ',concat(CODE, msg));
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
        'rep_op_ord_price_for_warning_samples',
        START_TIME,
        END_TIME,
        '0',
        @CODE_DESC
    );

END IF;

END
;
