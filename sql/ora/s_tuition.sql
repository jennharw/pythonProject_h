select
/*
 SCH110_KEDI_SCH_DIV,
SCH110_SCH_NM ,
sch214_year,
sch214_term,
sch214_std_id,
sch214_sch_cd,
sch214_grad_cd,
sch214_Dept_cd,
*/
sch214_std_id,
SUM(sch214_tuition_fee) + SUM(sch214_admission_fee) as 등록금장학 ,
SUM(SCH214_ETC_FEE)  AS ETC_장학
/*
sch110_sch_cd,
sch110_dpt_cd,
sch110_sch_div,
f_ncom_code_nm('SCH001',sch110_sch_div) ,
sch110_sponsor_cd,
sch110_sch_nm,
SCH110_SCH_NM_ENG,
sch110_tuition_pct,
, SCH110_RSN_DIV AS 지급사유코드
, F_NCOM_CODE_NM('SCH004',SCH110_RSN_DIV) AS 지급사유
, F_NCOM_CODE_NM('SCH011',SCH110_ORG_DIV) AS 장학금출처
, SCH110_REG_DIV --장학금재원구분(1:등록금2:일반)
, SCH110_LOAN_DIV --장학금융자구분(1:장학금2:융자)
, SCH110_ASSIST_YN --조교장학금여부(Y/N)
, F_NCOM_CODE_NM('SCH017',SCH110_OUT_RSN_DIV) AS 교외장학금지급사유
, F_NCOM_CODE_NM('SCH027',SCH110_GIVING_DIV) AS 지급형태구분
, SCH110_HAPSAN_DIV
, SCH110_KEDI_INSCH
, F_NCOM_CODE_NM('SCH032',SCH110_NATION_SCH_CD) AS 국가장학금코드
, F_NCOM_CODE_NM('SCH037',SCH110_KEDI_SCH_DIV) AS KEDI장학금구분
, F_NCOM_CODE_NM('NPER004',SCH110_ASSIST_DIV) 조교구분
*/
from sch214tl , sch110tl, kupid.COM020TL
where 1=1
and sch214_receive_yn = 'Y'
and sch110_sch_cd = sch214_sch_cd
and sch214_year > '2017'
and SUBSTR(SCH214_STD_ID,1,4) > '2017'
and sch110_sch_div ='1' --교외(2) / 교내(1)
AND SCH110_DPT_CD = COM020_DPT_CD
AND COM020_DIV IN ('131','132') --대학원
--AND sch214_std_id = '2017020887'
/*
AND SCH110_SCH_NM NOT LIKE '일반장학금%'
AND SCH110_SCH_NM NOT LIKE '신입생%'
AND SCH110_SCH_NM NOT LIKE '외국인장학%'
*/
--AND SCH110_SCH_NM LIKE '%연구%'
--AND SCH110_SCH_NM_ENG LIKE '%RESEARCH%'
--13126
--AND SCH110_SCH_CD IN ('13570','13126')
--AND SCH110_KEDI_SCH_DIV = '201' --성적우수
/*
AND SCH110_KEDI_SCH_DIV NOT IN (
'202' --저소득
,'203' --근로
,'204' --교직원
)

AND SCH110_RSN_DIV NOT IN (
'02' --가계곤란
,'06' --근로장학
,'08' --교직원
)
*/

--and SCH110_SPONSOR_CD = SCH130_SPONSOR_CD(+)
and (nvl2(SCH110_CRE_DT,to_char(SCH110_CRE_DT,'yyyy'),'1900') > '2017' or  SCH110_USE_YN ='Y')
GROUP BY SCH214_STD_ID

UNION all


select
/*
 SCH110_KEDI_SCH_DIV,
SCH110_SCH_NM ,
sch214_year,
sch214_term,
sch214_std_id,
sch214_sch_cd,
sch214_grad_cd,
sch214_Dept_cd,
*/
sch212_std_id,
SUM(sch212_tuition_fee) + SUM(sch212_admission_fee) as 등록금장학,
SUM(SCH212_ETC_FEE)  AS ETC_장학
/*
sch110_sch_cd,
sch110_dpt_cd,
sch110_sch_div,
f_ncom_code_nm('SCH001',sch110_sch_div) ,
sch110_sponsor_cd,
sch110_sch_nm,
SCH110_SCH_NM_ENG,
sch110_tuition_pct,
, SCH110_RSN_DIV AS 지급사유코드
, F_NCOM_CODE_NM('SCH004',SCH110_RSN_DIV) AS 지급사유
, F_NCOM_CODE_NM('SCH011',SCH110_ORG_DIV) AS 장학금출처
, SCH110_REG_DIV --장학금재원구분(1:등록금2:일반)
, SCH110_LOAN_DIV --장학금융자구분(1:장학금2:융자)
, SCH110_ASSIST_YN --조교장학금여부(Y/N)
, F_NCOM_CODE_NM('SCH017',SCH110_OUT_RSN_DIV) AS 교외장학금지급사유
, F_NCOM_CODE_NM('SCH027',SCH110_GIVING_DIV) AS 지급형태구분
, SCH110_HAPSAN_DIV
, SCH110_KEDI_INSCH
, F_NCOM_CODE_NM('SCH032',SCH110_NATION_SCH_CD) AS 국가장학금코드
, F_NCOM_CODE_NM('SCH037',SCH110_KEDI_SCH_DIV) AS KEDI장학금구분
, F_NCOM_CODE_NM('NPER004',SCH110_ASSIST_DIV) 조교구분
*/
from sch212tl , sch110tl, kupid.COM020TL
where 1=1
and sch212_receive_yn = 'Y'
and sch110_sch_cd = sch212_sch_cd
and sch212_year > '2017'

and SUBSTR(SCH212_STD_ID,1,4) > '2017'
and sch110_sch_div ='1' --교외(2) / 교내(1)
AND SCH110_DPT_CD = COM020_DPT_CD
AND COM020_DIV IN ('131','132') --대학원
--AND sch214_std_id = '2017020887'
/*
AND SCH110_SCH_NM NOT LIKE '일반장학금%'
AND SCH110_SCH_NM NOT LIKE '신입생%'
AND SCH110_SCH_NM NOT LIKE '외국인장학%'
*/
--AND SCH110_SCH_NM LIKE '%연구%'
--AND SCH110_SCH_NM_ENG LIKE '%RESEARCH%'
--13126
--AND SCH110_SCH_CD IN ('13570','13126')
--AND SCH110_KEDI_SCH_DIV = '201' --성적우수
/*
AND SCH110_KEDI_SCH_DIV NOT IN (
'202' --저소득
,'203' --근로
,'204' --교직원
)

AND SCH110_RSN_DIV NOT IN (
'02' --가계곤란
,'06' --근로장학
,'08' --교직원
)
*/

--and SCH110_SPONSOR_CD = SCH130_SPONSOR_CD(+)
and (nvl2(SCH110_CRE_DT,to_char(SCH110_CRE_DT,'yyyy'),'1900') > '2017' or  SCH110_USE_YN ='Y')
GROUP BY SCH212_STD_ID