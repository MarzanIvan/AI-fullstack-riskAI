<?php
    global$Server;
	require_once("../Declarations/Classes.php");
	$Server = new Server("ParfumTim",'mysql_db',"admin_user","admin_password");
	$Server->ConnectServer();
	$Server->ConnectDB('ParfumTim');
	$Server->SetCharset("UTF8");
?>