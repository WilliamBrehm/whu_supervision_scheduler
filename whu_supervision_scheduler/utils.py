"""
Utility functions for the WHU Supervision Scheduler.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

def validate_excel_file(filename: str) -> Tuple[bool, List[str]]:
    """
    Validate that the Excel file has all required sheets and columns.
    
    Args:
        filename: Path to the Excel file
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    try:
        # Check if file exists and can be read
        excel_file = pd.ExcelFile(filename)
        
        # Check required sheets
        required_sheets = ['Chairs', 'Exams', 'Blocked', 'Fixed']
        missing_sheets = [sheet for sheet in required_sheets 
                         if sheet not in excel_file.sheet_names]
        
        if missing_sheets:
            errors.append(f"Missing required sheets: {', '.join(missing_sheets)}")
            
        # Check Chairs sheet columns
        if 'Chairs' in excel_file.sheet_names:
            chairs_df = pd.read_excel(filename, sheet_name='Chairs')
            required_chair_cols = ['Chair', 'Share', 'Remote']
            missing_cols = [col for col in required_chair_cols 
                           if col not in chairs_df.columns]
            if missing_cols:
                errors.append(f"Chairs sheet missing columns: {', '.join(missing_cols)}")
                
        # Check Exams sheet columns
        if 'Exams' in excel_file.sheet_names:
            exams_df = pd.read_excel(filename, sheet_name='Exams')
            required_exam_cols = ['Start', 'End', 'Supervisors']
            missing_cols = [col for col in required_exam_cols 
                           if col not in exams_df.columns]
            if missing_cols:
                errors.append(f"Exams sheet missing columns: {', '.join(missing_cols)}")
                
        # Check Blocked sheet columns
        if 'Blocked' in excel_file.sheet_names:
            blocked_df = pd.read_excel(filename, sheet_name='Blocked')
            required_blocked_cols = ['Supervision Id', 'Exam Name', 'Chair', 'Start', 'End']
            missing_cols = [col for col in required_blocked_cols 
                           if col not in blocked_df.columns]
            if missing_cols:
                errors.append(f"Blocked sheet missing columns: {', '.join(missing_cols)}")
                
        # Check Fixed sheet columns
        if 'Fixed' in excel_file.sheet_names:
            fixed_df = pd.read_excel(filename, sheet_name='Fixed')
            required_fixed_cols = ['Chair', 'Supervision Id']
            missing_cols = [col for col in required_fixed_cols 
                           if col not in fixed_df.columns]
            if missing_cols:
                errors.append(f"Fixed sheet missing columns: {', '.join(missing_cols)}")
                
    except FileNotFoundError:
        errors.append(f"File not found: {filename}")
    except Exception as e:
        errors.append(f"Error reading file: {str(e)}")
        
    return len(errors) == 0, errors

