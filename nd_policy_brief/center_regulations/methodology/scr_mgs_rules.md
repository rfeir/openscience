**SCR and MGS Historical Coding Rules**  
**Scope**  
We study ordinary licensed, full-day, center-based childcare in all 50 states and Washington, D.C., as of July 1 in 2013, 2018, and 2023.  
We code two target ages:  
- **Infant:** exactly 6 months old  
- **Preschool:** exactly 4 years old  
We exclude family childcare homes, special programs, waivers, temporary staffing exceptions, and other nonstandard conditions. We use the ordinary center category for each state and year. For Louisiana, we use Type III where applicable. For Massachusetts, we use ordinary full-day preschool rules.  
**Staff-Child Ratio (SCR)**  
SCR is the number of children per required staff member:  
`SCR = children / required staff`  
For example, one staff member for five children gives an SCR of 5; two staff members for 11 children gives 5.5.  
We code the highest legally permitted SCR for the exact target age. We consider all applicable staffing tiers and do not round fractional ratios.  
**Mixed-Age Groups**  
We use a mixed-age provision only if:  
1. It applies even when a majority of children are in the target age category.  
2. It does not require stricter staff-child ratios for older children than their age-specific rules.  
Otherwise, we use the age-specific rule. We evaluate SCR and MGS separately.  
**Maximum Group Size (MGS)**  
MGS is a direct numerical limit on the number of children in a group or classroom. We do not infer MGS from staff-child ratios, room area, facility capacity, or building occupancy.  
We code the highest legally permitted direct group-size limit, applying the mixed-age rules above. SCR and MGS may come from different arrangements.  
If no direct MGS limit applies:  
```  
mgs = 0  
 no_mgs = 1  
 mgs_ratio = 0  
   
```  
Otherwise:  
```  
no_mgs = 0  
 mgs_ratio = mgs / scr  
   
```  
