SELECT
rec014_std_id
-- rec014_ent_year,
-- rec014_ent_term,
-- --rec014_grd_year,
-- --rec014_grd_term,
-- f_ncom_dept_nm(rec014_dept_cd),
-- f_ncom_code_nm('COM013',rec014_deg_div),
-- f_ncom_code_nm('REC051',rec014_rec_sts),
-- CASE WHEN REC014_ENT_TERM = '2R' AND REC014_GRD_TERM = '1R' THEN (REC014_GRD_YEAR - REC014_ENT_YEAR)*2 -1
-- WHEN REC014_ENT_TERM = '1R' AND REC014_GRD_TERM = '1R' THEN (REC014_GRD_YEAR - REC014_ENT_YEAR)*2
-- WHEN REC014_ENT_TERM = '1R' AND REC014_GRD_TERM = '2R' THEN (REC014_GRD_YEAR - REC014_ENT_YEAR)*2 +1
-- ELSE (REC014_GRD_YEAR - REC014_ENT_YEAR)*2
-- END AS PERIOD
from rec014tl, COM020TL a
where 1=1
and rec014_grad_cd = '0309'
and rec014_ent_year > '2017'
and rec014_dept_cd = a.com020_dpt_cd
and a.com020_campus =1

union

SELECT
rec012_std_id
-- rec012_ent_year,
-- REC012_ENT_TERM,
-- f_ncom_dept_nm(rec012_dept_cd),
-- f_ncom_code_nm('COM013',rec012_deg_div),
-- f_ncom_code_nm('REC051',rec012_rec_sts),
-- CASE WHEN REC012_ENT_TERM = '2R' THEN (2021 - REC012_ENT_YEAR)*2 -1
-- ELSE (2021 - REC012_ENT_YEAR)*2
-- END AS PERIOD
from rec012tl , COM020TL a
where 1=1
and rec012_grad_cd = '0309'
and rec012_ent_year > '2017'
and rec012_dept_cd = a.com020_dpt_cd
and a.com020_campus =1