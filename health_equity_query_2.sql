WITH visit_summary AS (
  SELECT
    person_id,
    MIN(visit_start_date) AS first_visit,
    MAX(visit_end_date) AS last_visit,
    COUNT(*) AS total_visits
  FROM visit_occurrence
  GROUP BY person_id
),

visits_per_year AS (
  SELECT
    person_id,
    ROUND(total_visits / GREATEST(DATEDIFF(last_visit, first_visit) / 365.25, 1), 2) AS avg_visits_per_year
  FROM visit_summary
),

total_visits AS (
  SELECT
    person_id,
    COUNT(*) AS total_encounters
  FROM visit_occurrence
  GROUP BY person_id
),

emergency_visits AS (
  SELECT
    person_id,
    COUNT(*) AS total_emergency_encounters
  FROM visit_occurrence
  WHERE visit_concept_id = 9203  -- Make sure this matches your emergency visit concept_id
  GROUP BY person_id
),

emergency_ratio_calc AS (
  SELECT 
    t.person_id,
    t.total_encounters,
    COALESCE(e.total_emergency_encounters, 0) AS total_emergency_encounters,
    ROUND(
      CASE
        WHEN t.total_encounters = 0 THEN NULL
        ELSE COALESCE(e.total_emergency_encounters, 0) * 1.0 / t.total_encounters
      END, 3
    ) AS emergency_ratio
  FROM total_visits t
  LEFT JOIN emergency_visits e ON t.person_id = e.person_id
),

resolved_flag AS (
  SELECT 
    co.person_id,
    MAX(CASE 
        WHEN c.concept_name = 'Resolved condition' THEN 1
        ELSE 0
    END) AS condition_resolved
  FROM condition_occurrence co
  JOIN concept c ON co.condition_status_concept_id = c.concept_id
  WHERE co.condition_status_concept_id IS NOT NULL
  GROUP BY co.person_id
)

SELECT
  v.person_id,
  v.avg_visits_per_year,
  e.emergency_ratio,
  COALESCE(r.condition_resolved, 0) AS condition_resolved,
  p.gender_concept_id,
  p.race_concept_id,
  p.ethnicity_concept_id,
  p.zip_code_source_value
FROM visits_per_year v
LEFT JOIN emergency_ratio_calc e ON v.person_id = e.person_id
LEFT JOIN resolved_flag r ON v.person_id = r.person_id
LEFT JOIN person p ON v.person_id = p.person_id
ORDER BY RAND();
