select rec012_std_id,  gpa032_mark1 as 성적
from gpa032tl g, rec012tl where gpa032_std_id = rec012_std_id
and rec012_grad_cd = '0309'
and rec012_ent_Year > '2017'
and gpa032_year = '0000'
union
select rec014_std_id,  gpa034_mark1 as 성적
from gpa034tl g, rec014tl where gpa034_std_id = rec014_std_id
and rec014_grad_cd = '0309'
and rec014_ent_Year > '2017'
and gpa034_year = '0000'