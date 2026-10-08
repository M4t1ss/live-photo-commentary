!macro customUninstall
  MessageBox MB_YESNO "Remove the app's data (settings, prompt sets and uploaded avatar models)?$\r$\nYou can skip this if you plan to reinstall." IDNO done
    RMDir /r "$APPDATA\io.github.m4t1ss.live-photo-commentary"
    RMDir /r "$LOCALAPPDATA\io.github.m4t1ss.live-photo-commentary"
  done:
!macroend
