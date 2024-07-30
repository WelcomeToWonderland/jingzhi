
from modules.line_value_pair_module import run
def process_scale_features(consolidated_dict):
    # Initialize the LineValuePairModule
    # line_value_module = LineValuePairModule(None, None, name='line_value_pair_module')
    # line_value_module.awake()  # Assuming the module needs to be awakened before using

    # Iterate through each image path in the consolidated_dict
    for img_path, data in consolidated_dict.items():
        # Prepare the input package for the LineValuePairModule
        input_package = {
            'splited_line_list': data['split_lines'],
            'filtered_text_list': data['ocr_result']
        }

        # Run the LineValuePairModule to get the output package
        output_package = run(input_package)

        # Update the consolidated_dict with the new scale_feature_list
        data['scale_feature_list'] = output_package['scale_feature_list']
        # data['text_feature_list'] = output_package['text_feature_list']  # If also needed


    return consolidated_dict
