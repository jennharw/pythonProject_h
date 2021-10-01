select
rec012_std_id,
f_ncom_code_nm('REC051',rec012_rec_sts),
--f_ncom_dept_nm(rec012_dept_cd),
--rec012_ent_year,
--rec012_ent_term,
--rec042_chg_cd,
f_ncom_code_nm('REC052',rec042_chg_Cd) as chg_nm,
--rec042_chg_rsn,
--rec042_std_id,
--rec042_year,
--rec042_term,
--rec042_begin_dt,
CASE WHEN REC012_ENT_TERM = '2R' AND REC042_TERM = '1R' THEN (REC042_YEAR - REC012_ENT_YEAR)*2 -1 +1
WHEN REC012_ENT_TERM = '1R' AND REC042_TERM = '1R' THEN (REC042_YEAR - REC012_ENT_YEAR)*2 +1
WHEN REC012_ENT_TERM = '1R' AND REC042_TERM = '2R' THEN (REC042_YEAR - REC012_ENT_YEAR)*2 +1 +1
ELSE (REC042_YEAR - REC012_ENT_YEAR)*2 +1
END AS period
--,rec042_seq,
--rec042_school_term
from  rec012tl d,rec042tl c
where rec042_cancel_div is null
and rec012_std_id = rec042_std_id(+)
--and rec012_std_id = '2016010797'
and rec012_grad_cd = '0309'
and rec012_ent_year > '2017'
and rec042_chg_cd in ('416','418','419','413','343','417','412','415','181','414','411','600')