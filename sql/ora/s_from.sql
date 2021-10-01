select
rec511_std_id,
eex600_docu_gpa+
eex600_oral_gpa as 입학성적,
--eex600_ent_yn,
--eex600_add_ent_yn,
--eex600_candidate_yn,
--eex300_type_cd,
--eex300_deg_cor,
--eex300_gra_sub_cd,
--eex300_gra_std_cd,
eex300_org_uni_cd as 학부출신,
--eex300_org_uni_sub_cd,
--eex300_org_uni_std_cd,
--eex300_org_uni_schd_dt,
--eex300_uni_gpa_ful_mark,
--eex300_uni_gpa_get_mark,
eex300_org_gra_Cd as 석사출신
--eex300_org_gra_sub_cd,
--eex300_org_gra_Std_cd,
--eex300_org_gra_schd_dt,
--eex300_job_yn,
--eex300_job_div_cd,
--eex300_job_nm,
/* 외국대학
eex300_org_uni_col_cd,
eex300_org_gra_col_cd
*/
--eex300_research_yn,
--eex300_schl_address,
--eex300_job_loca_nm
from rec511vw, eex600tl b , eex300tl c
where eex600_year = rec511_ent_year
and eex600_col_cd = rec511_ent_term
and eex600_schl_seq = rec511_std_id
and eex300_year = eex600_year
and eex300_col_cd = eex600_col_cd
and eex600_schl_num = eex300_schl_num
--and eex300_gra_cd = eex600_gra_cd
--and eex300_reg_num = eex600_reg_num
and rec511_ent_year > '2017'
and eex300_gra_cd = '0309'