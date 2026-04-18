PATCH FULL - COPY INTO C:\DHT elev\smartelevatormonitor

Day la GOI DAY DU tat ca cac file da duoc chinh sua de copy-de len project.
Muc tieu cuoi cung:
- UI clean V999 o frontend
- thong so hien o panel ben phai
- camera_session restore lai bbox features (nhan vien / chai nuoc / posture / action / last_event)

THU TU COPY DE TRANH NHAM:
1) Giai nen zip nay.
2) Copy TAT CA noi dung ben trong thu muc goc vao:
   C:\DHT elev\smartelevatormonitor
3) Dong backend/frontend dang chay.
4) Xoa cache frontend:
   Remove-Item -Recurse -Force .\node_modules\.cache -ErrorAction SilentlyContinue
5) Chay lai backend:
   cd "C:\DHT elev\smartelevatormonitor\backend"
   & "C:\DHT elev\smartelevatormonitor\venv\Scripts\python.exe" app.py
6) Chay lai frontend:
   cd "C:\DHT elev\smartelevatormonitor"
   npm start

KIEM TRA DUNG BAN:
- Frontend phai co badge: CAMERA AI CLEAN V999
- Backend camera_session phai co marker: BBOX_FEATURES_RESTORED
- Tren anh KHONG co HUD tong (FPS/People/Real time)
- Tren bbox CO the hien ID/ten, pose, bottle khi detect duoc

NEU MUON KIEM TRA NHANH:
Select-String -Path "C:\DHT elev\smartelevatormonitor\src\Component\camera\CameraPanel.js" -Pattern "CAMERA AI CLEAN V999"
Select-String -Path "C:\DHT elev\smartelevatormonitor\backend\existing_core\camera_session.py" -Pattern "BBOX_FEATURES_RESTORED"
