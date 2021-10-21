SELECT sum(휴학) 휴학횟수,
       sum(rec044_period) 휴학기간 ,sum(자퇴), REC044_sTD_ID AS std_id
from
(select decode(rec044_chg_cd, '211', 1, '212', 1) 휴학 ,rec044_period,
case when  rec044_chg_cd in ('416','418','419','413','343','417','412','415','181','414','411','600') then 1  end as 자퇴,
rec044_std_id
from rec044tl , rec014tl
where 1=1
and rec044_std_id = rec014_Std_id
and rec044_cancel_div is null
AND rec044_chg_cd in ('212', '211','416','418','419','413','343','417','412','415','181','414','411','600')
and rec014_grad_cd = '0309'
and rec014_ent_year > '2017'

UNION all
select decode(rec042_chg_cd, '211', 1, '212', 1) 휴학 ,rec042_period,
case when  rec042_chg_cd in ('416','418','419','413','343','417','412','415','181','414','411','600') then 1  end as 자퇴,
rec042_std_id
from rec042tl , rec012tl
where 1=1
and rec042_std_id = rec012_Std_id
and rec042_cancel_div is null
AND rec042_chg_cd in ('212', '211','416','418','419','413','343','417','412','415','181','414','411','600')
and rec012_grad_cd = '0309'
and rec012_ent_year > '2017'
 ) group by rec044_std_id