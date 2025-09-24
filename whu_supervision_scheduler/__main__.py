# whu_supervision_scheduler/__main__.py

"""
Main script for running the WHU Supervision Scheduler.

Usage:
    python -m whu_supervision_scheduler [options]
"""

import argparse
import sys
from .scheduler import WHUSupervisionScheduler
from .utils import validate_excel_file


def main():
    """Main function to run the scheduler from command line."""
    parser = argparse.ArgumentParser(
        description='WHU Supervision Scheduler - Assign exam supervision duties fairly'
    )
    
    # Required arguments
    parser.add_argument('input_file', 
                       help='Excel file containing supervision data')
    
    # Optional arguments
    parser.add_argument('-o', '--output', 
                       default='supervision_schedule.xlsx',
                       help='Output filename for the schedule (default: supervision_schedule.xlsx)')
    
    parser.add_argument('-t', '--timeout', 
                       type=int, 
                       default=60,
                       help='Solver timeout in seconds (default: 60)')
    
    parser.add_argument('-d', '--discount', 
                       type=float, 
                       default=0.15,
                       help='Discount factor for non-remote chairs (default: 0.15)')
    
    # Weight arguments
    parser.add_argument('--weight-weeks', 
                       type=int, 
                       default=40,
                       help='Weight for minimizing week indicators (default: 40)')
    
    parser.add_argument('--weight-days', 
                       type=int, 
                       default=10,
                       help='Weight for minimizing day indicators (default: 10)')
    
    parser.add_argument('--weight-consecutive-weeks', 
                       type=int, 
                       default=30,
                       help='Weight for encouraging consecutive weeks (default: 30)')
    
    parser.add_argument('--weight-consecutive-days', 
                       type=int, 
                       default=10,
                       help='Weight for encouraging consecutive days (default: 10)')
    
    parser.add_argument('--weight-assignment-fairness', 
                       type=int, 
                       default=90,
                       help='Weight for assignment count fairness (default: 90)')
    
    parser.add_argument('--weight-minutes-fairness', 
                       type=int, 
                       default=1,
                       help='Weight for time fairness (default: 1)')
    
    # Additional options
    parser.add_argument('--detailed', 
                       action='store_true',
                       help='Export detailed schedule with multiple sheets')
    
    parser.add_argument('--validate-only', 
                       action='store_true',
                       help='Only validate the input file without scheduling')
    
    args = parser.parse_args()
    
    # Validate input file
    print(f"Validating input file: {args.input_file}")
    is_valid, errors = validate_excel_file(args.input_file)
    
    if not is_valid:
        print("Input file validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
        
    print("Input file validation successful.")
    
    if args.validate_only:
        sys.exit(0)
    
    try:
        # Create scheduler instance
        scheduler = WHUSupervisionScheduler(
            weight_weeks=args.weight_weeks,
            weight_days=args.weight_days,
            weight_consecutive_weeks=args.weight_consecutive_weeks,
            weight_consecutive_days=args.weight_consecutive_days,
            weight_assignment_fairness=args.weight_assignment_fairness,
            weight_minutes_fairness=args.weight_minutes_fairness,
            not_remote_discount=args.discount,
            solver_timeout=args.timeout
        )
        
        # Read input data
        scheduler.read_supervision_file(args.input_file)
        
        # Print summary
        scheduler.print_summary()
        
        # Run scheduling
        success = scheduler.schedule()
        
        if success:
            # Print solution
            scheduler.print_solution()
            
            # Write basic output
            scheduler.write_solution_to_file(args.output)
            
        else:
            print("\nScheduling failed - no feasible solution found.")
            print("Consider:")
            print("  - Checking blocked time constraints")
            print("  - Verifying fixed assignments are feasible")
            print("  - Ensuring enough chairs for peak exam periods")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nError during scheduling: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_example():
    """Run an example scheduling scenario."""
    print("Running example scheduling scenario...")
    
    # Create scheduler with custom weights
    scheduler = WHUSupervisionScheduler(
        weight_weeks=50,  # Higher weight to minimize weeks for remote chairs
        weight_days=15,
        weight_consecutive_weeks=40,
        weight_consecutive_days=15,
        weight_assignment_fairness=100,  # High priority on fairness
        weight_minutes_fairness=2,
        not_remote_discount=0.20,  # 20% discount for non-remote chairs
        solver_timeout=30
    )
    
    # Read example file
    try:
        scheduler.read_supervision_file('example/supervision_template.xlsx')
        scheduler.print_summary()
        
        # Schedule
        if scheduler.schedule():
            scheduler.print_solution()
            scheduler.write_solution_to_file('example_schedule.xlsx')
        else:
            print("Example scheduling failed.")
            
    except FileNotFoundError:
        print("Example file 'example/supervision_template.xlsx' not found.")
        print("Please ensure the file exists in the current directory.")


if __name__ == '__main__':
    # Check if running example mode
    if len(sys.argv) > 1 and sys.argv[1] == '--example':
        run_example()
    else:
        main()