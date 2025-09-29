import json
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright, Error, TimeoutError
except ImportError:
    print("Error: Playwright library is not installed.")
    print("Please run the following commands to install:")
    print("1. pip install playwright")
    print("2. playwright install")
    exit()


# DSL renderer for converting JSON-based DSL to HTML
class DslRenderer:
    """A renderer that converts DSL (Domain Specific Language) JSON to HTML."""
    
    def _find_data_fields_recursive(self, node: dict, fields_set: set):
        """
        Recursively find all data binding fields in the DSL tree.
        
        Args:
            node: Current DSL node being processed
            fields_set: Set to collect unique field names for data binding
        """
        if not isinstance(node, dict): 
            return
        if "params" in node and isinstance(node["params"], dict):
            for attr_config in node["params"].values():
                if isinstance(attr_config, dict) and attr_config.get("bindType") == "Data":
                    field = attr_config.get("bindField")
                    if field: 
                        fields_set.add(field)
        if "children" in node and isinstance(node["children"], list):
            for child_node in node["children"]: 
                self._find_data_fields_recursive(child_node, fields_set)

    def generate_mock_data(self, dsl_json_string: str) -> dict:
        """
        Generate mock data for all data binding fields found in the DSL.
        
        Args:
            dsl_json_string: JSON string containing the DSL definition
            
        Returns:
            Dictionary with mock data for each data field
        """
        dsl_tree = json.loads(dsl_json_string)
        data_fields = set()
        self._find_data_fields_recursive(dsl_tree, data_fields)
        mock_data = {}
        for field in data_fields:
            if '].' in field:
                continue
            simple_name = field.split('.')[-1]
            mock_data[field] = f"Sample{simple_name}"
        return mock_data

    def render(self, dsl_json_string: str, data: dict) -> str:
        """
        Render DSL JSON to HTML with data binding.
        
        Args:
            dsl_json_string: JSON string containing the DSL definition
            data: Dictionary containing data for binding
            
        Returns:
            HTML string representation of the DSL
        """
        dsl_tree = json.loads(dsl_json_string)
        return self._render_node_recursive(dsl_tree, data)

    def _render_node_recursive(self, node: dict, data: dict) -> str:
        """
        Recursively render a DSL node to HTML.
        
        Args:
            node: Current DSL node being processed
            data: Dictionary containing data for binding
            
        Returns:
            HTML string representation of the node
        """
        if not isinstance(node, dict) or "type" not in node: 
            return ""
            
        tag_name = node.get("name", "div")
        attributes = []
        
        if node.get("className"): 
            attributes.append(f'class="{node["className"]}"')
            
        text_content = ""
        
        # Process parameters and data binding
        if "params" in node and isinstance(node["params"], dict):
            for attr_name, attr_config in node["params"].items():
                bind_type = attr_config.get("bindType")
                value = ""
                
                if bind_type == "Static":
                    value = attr_config.get("value", "")
                elif bind_type == "Data":
                    bind_field = attr_config.get("bindField")
                    if bind_field: 
                        value = data.get(bind_field, "")
                        
                if attr_name == "textContent":
                    text_content = str(value)
                else:
                    attributes.append(f'{attr_name}="{str(value)}"')
                    
        # Process children nodes
        children_html = ""
        if "children" in node and isinstance(node["children"], list):
            for child_node in node["children"]: 
                children_html += self._render_node_recursive(child_node, data)
                
        attributes_str = " ".join(attributes)
        self_closing_tags = {"img", "br", "hr", "input", "meta", "link"}
        
        if tag_name in self_closing_tags:
            return f'<{tag_name} {attributes_str}>'
        else:
            return f'<{tag_name} {attributes_str}>{text_content}{children_html}</{tag_name}>'


def capture_component_screenshot(html_content: str, output_path: str, target_selector: str):
    """
    Capture screenshot of HTML content using Playwright, waiting for specific selector.

    Args:
        html_content: Complete HTML string to render
        output_path: Path where the screenshot will be saved
        target_selector: CSS selector of the target element to wait for and screenshot (e.g., '#component-root')
    """
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_viewport_size({"width": 800, "height": 600})

        # Load content without relying on wait_until="networkidle"
        page.set_content(html_content)

        # Set reasonable timeout of 15 seconds
        try:
            component_element = page.locator(target_selector)
            component_element.wait_for(state="visible", timeout=15000)

            # Wait briefly for animations or final styling to complete
            page.wait_for_timeout(900)

            component_element.screenshot(path=output_path)
        finally:
            browser.close()


def process_and_screenshot_task(renderer: DslRenderer, dsl_data, output_path, html_name):
    """
    Core task function for processing DSL and capturing screenshots.
    
    Args:
        renderer: DslRenderer instance for DSL processing
        dsl_data: DSL data (string or dict with 'dsl_code' and 'mock_data')
        output_path: Path where the screenshot will be saved
        html_name: Name for the intermediate HTML file
        
    Returns:
        String indicating success or failure message
    """
    try:
        # Handle different data formats
        if type(dsl_data) is dict:
            dsl_code = dsl_data['dsl_code']
            mock_data = dsl_data['mock_data'] if 'mock_data' in dsl_data else None

            if mock_data is None:
                processed_mock_data = renderer.generate_mock_data(dsl_code)
            else:
                processed_mock_data = json.loads(mock_data)
                
            component_html = renderer.render(dsl_code, processed_mock_data)
        else:
            processed_mock_data = renderer.generate_mock_data(dsl_data)
            component_html = renderer.render(dsl_data, processed_mock_data)
            
        # Add unique ID for target component in HTML template
        target_id = "component-root"

        full_html_for_screenshot = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body style="background-color: transparent; margin: 0; padding: 0;">
            <!-- Added target ID for component selection -->
            <div id="{target_id}" style="display: inline-block;">
                {component_html}
            </div>
        </body>
        </html>
        """
        
        # Save HTML file for reference
        with open(html_name, 'w', encoding='utf-8') as f:
            f.write(full_html_for_screenshot)

        # Capture screenshot with updated function
        capture_component_screenshot(
            html_content=full_html_for_screenshot,
            output_path=output_path,
            target_selector=f"#{target_id}"
        )
        return f"Success: {Path(output_path).name}"
    except TimeoutError:
        return f"Failed (Timeout): {Path(output_path).name} - Element did not appear within timeout."
    except Exception as e:
        # Catch all other errors
        return f"Failed (Error): {Path(output_path).name} with error: {repr(e).__name__}: {str(e)}"


# Main execution block
if __name__ == "__main__":
    # Initialize renderer
    renderer = DslRenderer()
    
    # Sample DSL code for testing
    dsl_code = '{"type":"Tag","name":"div","className":"w-full p-4 bg-white rounded-lg shadow-md flex flex-col items-center space-y-4","children":[{"type":"Tag","name":"h1","className":"text-xl font-bold text-gray-800 mb-2","params":{"textContent":{"bindType":"Static","value":"You can borrow"}}},{"type":"Tag","name":"span","className":"inline-block px-2 py-1 text-sm font-semibold leading-tight text-blue-600 bg-blue-100 rounded-full","params":{"textContent":{"bindType":"Static","value":"Credit protection enabled"}}},{"type":"Tag","name":"p","className":"text-4xl font-extrabold mt-4","params":{"textContent":{"bindType":"Data","bindField":"loanAmount"}}},{"type":"Tag","name":"p","className":"text-base text-gray-600","params":{"textContent":{"bindType":"Static","value":"Total limit "}},"children":[{"type":"Tag","name":"span","className":"font-semibold","params":{"textContent":{"bindType":"Data","bindField":"totalQuota"}}},{"type":"Tag","name":"span","className":"ml-1","params":{"textContent":{"bindType":"Static","value":", daily interest "}},"children":[{"type":"Tag","name":"span","className":"text-red-500 font-semibold","params":{"textContent":{"bindType":"Data","bindField":"dailyInterestRate"}}}]}]},{"type":"Tag","name":"div","className":"flex items-center bg-yellow-50 rounded-lg p-2 w-max mx-auto","children":[{"type":"Tag","name":"img","className":"w-6 h-6 mr-2","params":{"src":{"bindType":"Static","value":"<image>red packet icon</image>"}}},{"type":"Tag","name":"span","className":"text-blue-600 font-medium","params":{"textContent":{"bindType":"Static","value":"Multiple discounts available"}}}]},{"type":"Tag","name":"button","className":"bg-blue-600 text-white hover:bg-blue-700 active:bg-blue-800 rounded-3xl py-2 px-4 border-0 inline-flex items-center justify-center font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none w-full max-w-xs","params":{"textContent":{"bindType":"Static","value":"Apply for loan"}}},{"type":"Tag","name":"button","className":"bg-white text-gray-700 hover:bg-gray-50 active:bg-gray-100 rounded-3xl py-2 px-4 border border-gray-300 inline-flex items-center justify-center font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none w-full max-w-xs","params":{"textContent":{"bindType":"Static","value":"Repay loan"}},"children":[{"type":"Tag","name":"span","className":"block text-sm text-gray-500 mt-1","params":{"textContent":{"bindType":"Static","value":"Due today "}},"children":[{"type":"Tag","name":"span","className":"font-semibold","params":{"textContent":{"bindType":"Data","bindField":"todayRepaymentAmount"}}}]}]}]}'
    
    # Generate mock data or use custom data
    mock_data = renderer.generate_mock_data(dsl_code)
    
    # Prepare DSL data for processing
    dsl_data = {
        'dsl_code': dsl_code,
        'mock_data': json.dumps(mock_data)
    }
    
    # Process and capture screenshot
    res_message = process_and_screenshot_task(renderer, dsl_data, "render_ui.png", "render_ui.html")
    print(res_message)