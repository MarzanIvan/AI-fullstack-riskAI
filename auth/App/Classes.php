<?php

class Server{
	private $NameServer;
	private $NameRoot;
	private $Password;
	private $Database;
	public $DB_HANDLE;
	private $ConnectedDB;
	public function __construct($InputDatabase, $InputNameServer, $InputNameRoot, $InputPassword) {
		$this->Database = $InputDatabase;
		$this->NameServer = $InputNameServer;
		$this->NameRoot = $InputNameRoot;
		$this->Password = $InputPassword;
		$this->DB_HANDLE = NULL;
		$this->ConnectedDB = NULL;
	}

    public function get_DB_HANDLE() {
        return $this->DB_HANDLE;
    }

	public function ConnectServer() {
		try {
			$this->DB_HANDLE = mysqli_connect($this->NameServer, $this->NameRoot, $this->Password, $this->Database);
			if (!$this->DB_HANDLE) {
				throw new Exception();
			}
		}catch( Exception $ExceptionErrorConnecting ) {
			die($ExceptionErrorConnecting->getMessage());
		}
	}
    public function SendQuery($query) {
        return mysqli_query($this->DB_HANDLE, $query);
    }

    public function ConnectDB($NameDB) {
        $this->ConnectedDB = mysqli_select_db($this->DB_HANDLE, $NameDB) or die('The error connecting has been happened');
    }
    public function SetCharset( $InputCharset ) {
		try {
			$SettingCharset = mb_internal_encoding($InputCharset);
			if (!$SettingCharset) {
				throw new Exception("The trying to set the error charset");
			}
			$this->SendQuery("SET NAMES "."'".$InputCharset."'");
		}catch(Exception $ExceptionErrorConnecting) {
			die($ExceptionErrorConnecting->getMessage());
		}
	}
}

class DB{
	private $NameDB;
	private $Connection;
	public function __construct( $InputNameDB ) {
		$this->NameDB = $InputNameDB;
	}
	public function ConnectDB() {
		try {
			$connection = mysqli_select_db($this->NameDB);
			if (!$connection) {
				throw new Exception("The error connecting has been happened");
			}
		}catch(Exception $ExceptionErrorConnecting) {
			die($ExceptionErrorConnecting->getMessage());
		}
	}
	public function SetCharset( $InputCharset ) {
		try {
			$SettingCharset = mb_internal_encoding($InputCharset);
			if (!$SettingCharset) {
				throw new Exception("The trying to set the error charset");
			}
			mysqli_query("SET NAMES "."'".$InputCharset."'");
		}catch(Exception $ExceptionErrorConnecting) {
			die($ExceptionErrorConnecting->getMessage());
		}
	}
}


class ExceptionLoadingImage extends  Exception{
	public function ShowExplainImage() {
		?>
		echo 'error';
		<?
	}
}

class File {
	private $PathFile;
	public function __construct( $PathFile) {
		$this->PathFile = $PathFile;
	}

	public function SaveFileToPath( $PathToSave ) {
		move_uploaded_file( $this->PathFile, "../".$PathToSave );
	}

	public function ShowImage() {
		try {
			if ( file_exists($this->PathFile) ) {
				throw new ExceptionLoadingImage();
			}
			?>
				<img src="../<? echo $this->PathFile;?>">
			<?
		} 
		catch ( ExceptionLoadingImage $ErrorLoading ) {
			$ErrorLoading->ShowExplainImage();
		}
	}

	public function DeleteFile() {
		unlink( $this->PathFile );
	}

	public function __destruct() {
		$this->PathFile = null;
	} 

}

class ExceptionIncorrectInputting {
	private $MessageExplaining;

	public function __construct( $Message ) {
		$this->MessageExplaining = $Message;
	}

	public function __destruct() {
		$this->MessageExplaining = null;
	}

	public function ShowMessage() {
		echo $this->MessageExplaining;
	}

	public function LoadForm( $PathToForm ) {
		require_once($PathToForm);
	}
}

?>
