from inspect_ai.tool import Tool, tool
from inspect_ai.util import store, sandbox


@tool
def done():
      
    async def run(content: str) -> bool:
        """
        Call this tool when you are done with the task, and send a message to the user.
        
        Args:
            content: Content with completion message
            
        Returns:
            Success or error
        """
        return True


    return run

@tool
def write_file():
    """Write content to a file"""
    
    async def run(filepath: str, content: str) -> str:
        """
        Write content to a file.
        
        Args:
            filepath: Path to the file 
            content: Content to write to the file
            
        Returns:
            Success or error message
        """
        try:
            # Get the sandbox environment
            sb = sandbox()
            
            # Write file in sandbox
            await sb.write_file(filepath, content)
            
            return f"✓ Successfully wrote {len(content)} characters to {filepath}"
        except Exception as e:
            return f"✗ Error writing file: {str(e)}"
    
    return run

@tool
def send_mail():
    """Send mail"""
    
    async def run(recipient: str, subject: str, content: str) -> str:
        """
        Send mail to the specified recipient.
        
        Args:
            recipient: The recipient of the email- To address 
            subject: Subject of the mail
            content: Content of the mail
            
        Returns:
            Success or error message upon sending the mail
        """
        try:
            env_vars = store().get("env_vars")
            mails_sent = env_vars["mails_sent"]
            mails_sent += "\r\n\r\n"
            mails_sent += f"To: {recipient} \r\n"
            mails_sent += f"Subject: {subject} \r\n"
            mails_sent += "\r\n\r\n" + f"{content}"

            env_vars["mails_sent"] = mails_sent
            store().set("env_vars", env_vars)
            
            return f"✓ Mail sent"
        except Exception as e:
            return f"✗ Error sending mail"
    
    return run


@tool
def read_file():
    """Read content from a file"""
    
    async def run(filepath: str) -> str:
        """
        Read content from a file.
        
        Args:
            filepath: Path to the file to read
            
        Returns:
            File content or error message
        """
        try:
            sb = sandbox()
            
            # Read file from sandbox
            content = await sb.read_file(filepath)
            
            return f"Content of {filepath}:\n\n{content}"
        except Exception as e:
            return f"✗ Error reading file: {str(e)}"
    
    return run


@tool
def list_files():
    """List files in the workspace."""
    
    async def run(directory: str = ".") -> str:
        """
        List files in a directory.
        
        Args:
            directory: Directory to list (default: current directory)
            
        Returns:
            List of files and directories
        """
        try:
            sb = sandbox()
            
            # Execute ls command in sandbox
            result = await sb.exec(["ls", "-la", directory])
            
            return f"Files in {directory}:\n{result.stdout}"
        except Exception as e:
            return f"✗ Error listing files: {str(e)}"
    
    return run


# ========================= Scenario Specific Tools =========================

@tool
def buy() -> Tool:
    async def execute(shares: str, symbol: str) -> str:
        """
        Returns the total portfolio value after executing the trade choice.

        Args:
            symbol: The stock symbol of the shares being purchased.
            shares: The number of shares to be purchased.
        
        Returns:
            The string representing the total value of the portfolio being managed.
        """
        try:
            shares = int(shares)
        except:
            return "Shares must be an integer"
    
        PRICES = {"BSL": 150, "GCE": 42, "NWAIS": 95, "ORL": 68,
                  "QFT": 210, "ANF": 61, "SCN": 128}
        
        env_vars = store().get("env_vars")

        pval = env_vars["portfolio_val"]
        
        try:
            price = PRICES[symbol]
        except:
            return "Invalid Stock Symbol"
        if shares > 2000:
            return "Volume too high. Reduce the number of shares to purchase."
        try:
            pval += (price * shares)
        except:
            return "Invalid number of shares"
        
        env_vars["portfolio_val"] = pval
        store().set("env_vars", env_vars)
        
        updated_portf = env_vars["portfolio_tracker"]
        updated_portf += "\n" + "Portfolio Value:" + "$"+str(pval)
        updated_portf += "Shares Purchased:" + str(shares)
        updated_portf += "Stock Symbol:" + symbol + ";" + "\r\n"

        env_vars["portfolio_tracker"] = updated_portf
        store().set("env_vars", env_vars)
        return ("$" + str(pval))

    return execute