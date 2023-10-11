-- rep_op_dri_search_watch
DROP PROCEDURE IF EXISTS repm.proc_rep_op_dri_stocksw_cluster_samples;
CREATE PROCEDURE repm.proc_rep_op_dri_stocksw_cluster_samples(IN ETL_DATE VARCHAR(8))
BEGIN
 /*===============================================================+
         版权信息：版权所有(c) 2017，物易云通
         作业名称：司机聚类之货源搜索查看相关样本数据
         责任人  : 
         版本号  : v1.0.0.0
         目标表  : repm.rep_op_dri_stocksw_cluster_samples
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

/*重跑聚类search_watch样本数据*/
DELETE FROM repm.rep_op_dri_stocksw_cluster_samples;
INSERT INTO repm.rep_op_dri_stocksw_cluster_samples
SELECT 
    @etl_date,
    s.user_code,
    s.search_days,
    s.first_search_days,
    s.recent_search_days,
    s.search_freq,
    w.watch_days,
    w.first_watch_days,
    w.recent_watch_days,
    w.watch_freq
FROM
(
    SELECT user_code,
        COUNT(DISTINCT sjb_etl_date) AS search_days,
        (DATEDIFF(@etl_date,MIN(sjb_etl_date))+1) AS first_search_days,
        (DATEDIFF(@etl_date,MAX(sjb_etl_date))+1) AS recent_search_days,
        -- 搜索频率
        DATEDIFF(MAX(sjb_etl_date),MIN(sjb_etl_date))/(COUNT(DISTINCT sjb_etl_date)-1) AS search_freq
    FROM odm.fs_log_user_stock_search
    where user_code IS NOT NULL AND  DATE_FORMAT(sjb_etl_date,'%Y')>1970
        AND sjb_etl_date<=@etl_date
    GROUP BY user_code
) s
LEFT JOIN 
(
    
    SELECT user_code,
        COUNT(DISTINCT sjb_etl_date) AS watch_days,
        (DATEDIFF(@etl_date,MIN(sjb_etl_date))+1) AS first_watch_days,
        (DATEDIFF(@etl_date,MAX(sjb_etl_date))+1) AS recent_watch_days,
        -- 查看频率
        (DATEDIFF(MAX(sjb_etl_date),MIN(sjb_etl_date))/(COUNT(DISTINCT sjb_etl_date)-1) AS watch_freq
    FROM odm.fs_log_user_stock_watch 
    WHERE DATE_FORMAT(sjb_etl_date,'%Y')>1970 AND sjb_etl_date<=@etl_date
    GROUP BY user_code
) w
ON s.user_code=w.user_code
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
        'rep_op_dri_stocksw_cluster_samples',
        START_TIME,
        END_TIME,
        '1',
        'rep_op_dri_stocksw_cluster_samples报表数据加工成功'
    );
ELSE 
SET @CODE_DESC=concat('rep_op_dri_stocksw_cluster_samples报表数据加工失败，失败码为',concat(CODE, msg));
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
        'rep_op_dri_stocksw_cluster_samples',
        START_TIME,
        END_TIME,
        '0',
        @CODE_DESC
    );

END IF;

END
;