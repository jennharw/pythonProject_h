select user_pk1,
--course_pk1,
event_type,
date(foo.timestamp) date,
date_part('hour', foo.timestamp) as hour,
count(user_pk1) as count
from
    (
    select * from activity_accumulator aa
    where timestamp between '{}' and '{}'
    and event_type = 'LOGIN_ATTEMPT') as foo
group by
user_pk1,
event_type,
date(foo.timestamp),
date_part('hour', foo.timestamp)