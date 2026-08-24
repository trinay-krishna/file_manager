

from file_manager import store 

shelves = []

file_names = [ 
    "Global_warming.pdf",
    "homeowners_insurance_declaration.pdf",
    "lab_results_summary.pdf",
    "personal_loan_agreement.pdf",
    "vehicle_purchase_warranty.pdf",
    "mortgage_loan_commitment.pdf"
    ]

print("Uploading files")

for file_name in file_names:

    shelves = store(file_name=file_name, main_shelves = shelves)

    print(f"Updated shelves so far:\n{shelves}")


print("Final Shelves")
print(shelves)