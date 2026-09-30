# Azerbaijan KB — Quality Report

## Structural checks
- Total cards: **150**
- Category counts: **{'food': 25, 'sports': 25, 'family': 25, 'education': 25, 'holidays_celebrations_leisure': 25, 'work_life': 25}**
- Unique IDs: **150**
- JSONL parsing: **PASS**
- Required schema fields: **PASS**
- Direct source URL present: **PASS**
- Category target 25 each: **PASS**

## Source diversity
- Distinct source domains: **10**
- Domain counts: `{'ich.unesco.org': 9, 'www.unesco.az': 16, 'azerbaijan.travel': 20, 'edu.gov.az': 30, 'www.stat.gov.az': 25, 'stat.gov.az': 5, 'family.gov.az': 15, 'www.family.gov.az': 5, 'unesco.az': 14, 'dost.gov.az': 11}`
- Highest domain share: **20.0%**
- At least 2 domains in each category: **True**
- Source classes: `{'international_heritage': 25, 'tourism_board': 20, 'education_ministry': 30, 'statistical_agency': 30, 'government_agency': 31, 'cultural_institution': 14}`

## Quality caveats
- No human cultural verification was performed, per project constraint.
- Some regional food/leisure cards rely on Azerbaijan Travel; these are marked medium quality.
- Statistical cards describe documented institutional patterns rather than cultural norms.
- UNESCO heritage descriptions are strong evidence for documented traditions but should not be interpreted as claiming universal participation.
- The corpus intentionally avoids claiming exhaustive representation of Azerbaijan.

## Research coverage
Food, sport, family, education, holidays/celebrations/leisure and work-life were all populated from multiple institutional or authoritative sources. Regional evidence includes Lankaran, Shaki, Lahij and Nakhchivan, while national statistics provide broader geographic coverage.
